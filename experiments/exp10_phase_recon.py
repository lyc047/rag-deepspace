"""Phase 10: 相位对齐 + 模板重建 = 让 periodic/transient 也受益.

exp09 结论: 模板重建对 periodic/transient 无效 (残差 > 噪声)
原因: 相位偏移 / 事件时间错位

方案: 重建前先对齐 — y_template 对齐到 y_original, 再算残差
预期: periodic MSE 从 1.3 → <0.1, transient MSE 从 0.6 → <0.1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm
from scipy.stats import trim_mean

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import (
    HierarchicalRetriever, SignalOnlyRetriever,
    PhysicsOnlyRetriever,
)
from reconstruction import reconstruct_from_template
from preprocessing import phase_align_via_cross_correlation
from metrics import mse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reconstruct_with_alignment(y_orig, y_recv, retrieval_results, kb, channel):
    """相位对齐 + 模板重建."""
    if not retrieval_results:
        return y_recv

    best_id = retrieval_results[0][0]
    record = kb.get_record(best_id)
    if record is None:
        return y_recv

    y_template = record.raw_data

    # 对齐 template 到 original
    aligned, lag, corr = phase_align_via_cross_correlation(y_orig, y_template)
    if corr < 0.3:
        aligned = y_template  # 相关性太弱, 放弃对齐

    # 用对齐后的模板算残差
    residual = y_orig - aligned
    residual = channel.forward(residual)
    return aligned + residual


def run_single_signal(signal_type, n_templates=1000, n_test=40, snr_steps=7):
    """对比: 无对齐 vs 有对齐."""
    print(f"\n{'='*60}")
    print(f"Signal: {signal_type} (with phase-aligned reconstruction)")
    print(f"{'='*60}")

    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=42)

    cache = {
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Signal-Only': SignalOnlyRetriever(kb, normalize=False),
    }
    names = [
        'No Retrieval',
        'Signal-Only (no align)',
        'Signal-Only (aligned)',
        'Stats-Only (no align)',
        'Stats-Only (aligned)',
    ]

    snr_vals = np.linspace(-170, -130, snr_steps)
    raw = {n: {s: [] for s in snr_vals} for n in names}

    for snr_db in tqdm(snr_vals, desc=f'  {signal_type}'):
        for _ in range(n_test):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type, physics=physics,
                seed=np.random.randint(0, 2**31 - 1))
            y_orig = sample.signal
            ch = DeepSpaceChannel(distance_au=physics['distance_au'], snr_db=snr_db)
            y_recv = ch.forward(y_orig)
            qd, qa = physics['distance_au'], physics['sun_earth_probe_angle']

            for mname in names:
                if mname == 'No Retrieval':
                    raw[mname][snr_db].append(mse(y_orig, y_recv))
                    continue

                # Get template
                if 'Signal-Only' in mname:
                    rr = cache['Signal-Only'].retrieve(y_recv, k=1)
                elif 'Stats-Only' in mname:
                    h = HierarchicalRetriever(kb, coarse_k=n_templates, fine_k=3,
                        normalize=False, distance_window=3.0, angle_window=90.0,
                        stat_filter=True, stat_max_keep=200)
                    rr = h.retrieve(y_recv, distance_au=qd, sun_angle=qa)
                else:
                    continue

                if 'aligned' in mname:
                    y_recon = reconstruct_with_alignment(y_orig, y_recv, rr, kb, ch)
                else:
                    y_recon = reconstruct_from_template(y_recv, rr, kb, y_original=y_orig, channel=ch)

                raw[mname][snr_db].append(mse(y_orig, y_recon))

    results = {}
    for mname in names:
        results[mname] = [trim_mean(raw[mname][s], 0.05) for s in snr_vals]

    mid = len(snr_vals) // 2
    print(f"  SNR=-150dB:")
    for mname in names:
        print(f"    {mname:25s}: {results[mname][mid]:.4f}")

    if 'Signal-Only (no align)' in results and 'Signal-Only (aligned)' in results:
        impr = results['Signal-Only (no align)'][mid] / max(results['Signal-Only (aligned)'][mid], 1e-10)
        print(f"    Alignment improvement: {impr:.1f}x")

    return snr_vals, results


def plot_phase_recon(all_data, save_path='results/exp10_phase_recon.png'):
    """左右对比: periodic, transient 有无对齐."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {
        'No Retrieval': '#d32f2f',
        'Signal-Only (no align)': '#1976d2',
        'Signal-Only (aligned)': '#e65100',
        'Stats-Only (no align)': '#9c27b0',
        'Stats-Only (aligned)': '#2e7d32',
    }
    styles = {
        'No Retrieval': ('-', 1.5),
        'Signal-Only (no align)': ('--', 1.5),
        'Signal-Only (aligned)': ('-', 2.5),
        'Stats-Only (no align)': ('--', 1.5),
        'Stats-Only (aligned)': ('-', 2.5),
    }

    for idx, (signal_type, (snr_vals, results)) in enumerate(all_data.items()):
        ax = axes[idx]

        for mname in ['No Retrieval', 'Signal-Only (no align)',
                      'Signal-Only (aligned)', 'Stats-Only (aligned)']:
            if mname not in results:
                continue
            ls, lw = styles[mname]
            ax.semilogy(snr_vals, results[mname],
                       color=colors[mname], linestyle=ls, linewidth=lw,
                       marker='o', markersize=4, markevery=1,
                       alpha=0.9, label=mname)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('MSE (log scale)', fontsize=12)
        ax.set_title(signal_type, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25, which='both')
        ax.legend(fontsize=8, framealpha=0.9)

        # Improvement annotation
        no_align = results.get('Signal-Only (no align)', [0])
        aligned = results.get('Signal-Only (aligned)', [1])
        mid = len(snr_vals) // 2
        if no_align[mid] > aligned[mid]:
            ratio = no_align[mid] / max(aligned[mid], 1e-10)
            ax.annotate(f'{ratio:.0f}x better\nwith alignment',
                       xy=(snr_vals[mid], aligned[mid]),
                       xytext=(snr_vals[mid], aligned[mid] * 100),
                       fontsize=10, fontweight='bold', color='#e65100',
                       arrowprops=dict(arrowstyle='->', color='#e65100', lw=2),
                       ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=1000)
    p.add_argument('--test-samples', type=int, default=40)
    p.add_argument('--snr-steps', type=int, default=7)
    args = p.parse_args()

    all_data = {}
    for st in ['periodic', 'transient']:
        snr_vals, results = run_single_signal(
            st, n_templates=args.templates,
            n_test=args.test_samples, snr_steps=args.snr_steps)
        all_data[st] = (snr_vals, results)

    plot_phase_recon(all_data)
