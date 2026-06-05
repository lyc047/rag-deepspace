"""Phase 4: 多维相似度 + B级信道 + 多信号类型全量实验.

对比：
- 不同距离度量 (euclidean, multi)
- 不同信号类型 (slow_varying, periodic, transient, science_data)
- B级信道 (多普勒 + 太阳闪烁)
- 自适应窗口 v2
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm
from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import HierarchicalRetriever, SignalOnlyRetriever
from reconstruction import reconstruct_from_template
from metrics import mse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def adaptive_window_v2(sun_angle: float, snr_db: float = -150,
                       w_base: float = 1.0, w_max: float = 2.0) -> float:
    angle_factor = 0.5 * (sun_angle / 90.0)
    snr_norm = (snr_db + 170) / 40.0
    snr_factor = 0.5 * snr_norm
    return min(w_base + angle_factor + snr_factor, w_max)


def test_signal_type(signal_type, n_templates=300, n_test=50, n_snr=7,
                     channel_level='A'):
    """对单种信号类型运行完整对比."""
    print(f"\n{'='*60}")
    print(f"Signal: {signal_type:15s} | Channel: {channel_level} | "
          f"Templates: {n_templates}")
    print(f"{'='*60}")

    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=42)

    snr_values = np.linspace(-170, -130, n_snr)
    metrics_list = ['euclidean', 'multi']
    results = {}

    for metric in metrics_list:
        for strategy in ['signal-only', 'hier-adapt']:
            key = f'{strategy}_{metric}'
            results[key] = []

    for snr_db in tqdm(snr_values, desc=f"  {signal_type}"):
        mse_buf = {k: [] for k in results}

        for _ in range(n_test):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type=signal_type, physics=physics,
                seed=np.random.randint(0, 2**31 - 1),
            )
            y_orig = sample.signal

            ch = DeepSpaceChannel(
                distance_au=physics['distance_au'], snr_db=snr_db,
                doppler_hz=physics['doppler_shift_hz'],
                scintillation_index=physics['scintillation_index'],
                fs_hz=10.0,
            )
            y_recv = ch.forward(y_orig, level=channel_level)

            qd = physics['distance_au']
            qa = physics['sun_earth_probe_angle']
            q_snr = physics['snr_db']

            for metric in metrics_list:
                # Signal-Only
                so = SignalOnlyRetriever(kb, metric=metric, normalize=False)
                rr_so = so.retrieve(y_recv, k=1)
                y_so = reconstruct_from_template(y_recv, rr_so, kb)
                mse_buf[f'signal-only_{metric}'].append(mse(y_orig, y_so))

                # Hier-Adapt
                dw = adaptive_window_v2(qa, q_snr)
                da = min(90, dw * 30)
                hier = HierarchicalRetriever(
                    kb, coarse_k=200, fine_k=3,
                    metric=metric, normalize=False,
                    distance_window=dw, angle_window=da,
                )
                rr_h = hier.retrieve(y_recv, distance_au=qd, sun_angle=qa)
                y_h = reconstruct_from_template(y_recv, rr_h, kb)
                mse_buf[f'hier-adapt_{metric}'].append(mse(y_orig, y_h))

        for k in results:
            results[k].append(np.mean(mse_buf[k]))

    # 打印SNR=-150dB结果
    mid = len(snr_values) // 2
    print(f"\n  SNR=-150dB results:")
    for k in results:
        marker = ' ★' if 'adapt' in k and 'multi' in k else ''
        print(f"    {k:30s}: MSE = {results[k][mid]:.6f}{marker}")

    return results, snr_values


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=300)
    p.add_argument('--test-samples', type=int, default=40)
    p.add_argument('--channel', default='B')
    args = p.parse_args()

    signal_types = ['slow_varying', 'periodic', 'transient', 'science_data']
    all_results = {}

    for st in signal_types:
        res, snr = test_signal_type(
            st, n_templates=args.templates,
            n_test=args.test_samples,
            channel_level=args.channel,
        )
        all_results[st] = res

    # 汇总图表：4×2 子图（每种信号类型一行）
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, st in enumerate(signal_types):
        ax = axes[idx]
        res = all_results[st]
        snr_vals = np.linspace(-170, -130, 7)

        for key, mse_list in res.items():
            parts = key.split('_')
            strat = 'Hier-adapt' if 'adapt' in key else 'SignalOnly'
            met = 'multi' if 'multi' in key else 'euc'
            label = f'{strat}+{met}'
            ls = '-' if 'adapt' in key else '--'
            lw = 2.5 if ('adapt' in key and 'multi' in key) else 1.2
            color = '#7b1fa2' if ('adapt' in key and 'multi' in key) else \
                    '#1976d2' if 'adapt' in key else '#f57c00'
            ax.semilogy(snr_vals, mse_list, label=label, color=color,
                        linestyle=ls, linewidth=lw, marker='o', markersize=3)

        ax.set_title(f'{st}', fontsize=12)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('MSE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('results/exp04_multidimensional.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: results/exp04_multidimensional.png")


if __name__ == '__main__':
    main()
