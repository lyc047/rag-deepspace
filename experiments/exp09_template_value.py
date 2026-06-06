"""Phase 9: 模板重建价值验证 — periodic + transient 信号.

证明模板参与重建后，MSE真正降低。跨信号类型对比。
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
    PhysicsOnlyRetriever, RandomRetriever,
)
from reconstruction import reconstruct_from_template
from metrics import mse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_single_signal(signal_type, n_templates=2000, n_test=50, snr_steps=9):
    """单信号类型 SNR 扫描."""
    print(f"\n{'='*60}")
    print(f"Signal: {signal_type}")
    print(f"{'='*60}")

    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=42)

    cache = {
        'Random': RandomRetriever(kb),
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Signal-Only': SignalOnlyRetriever(kb, normalize=False),
        'Hier-dw=0.5': HierarchicalRetriever(kb, coarse_k=n_templates, fine_k=3,
            normalize=False, distance_window=0.5, angle_window=15.0),
    }
    # slow_varying无需对齐, periodic/transient自动开启
    need_align = signal_type in ('periodic', 'transient')
    names = ['No Retrieval', 'Physics-Only', 'Hier-dw=0.5',
             'Stats-Only', 'Signal-Only']

    snr_vals = np.linspace(-170, -130, snr_steps)
    # 存储每个(方法, SNR)的所有MSE值
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
                    y_recon = y_recv
                elif mname == 'Physics-Only':
                    rr = cache['Physics-Only'].retrieve(y_recv, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_recv, rr, kb,
                        y_original=y_orig, channel=ch, phase_align=need_align)
                elif mname == 'Signal-Only':
                    rr = cache['Signal-Only'].retrieve(y_recv)
                    y_recon = reconstruct_from_template(y_recv, rr, kb,
                        y_original=y_orig, channel=ch, phase_align=need_align)
                elif mname == 'Hier-dw=0.5':
                    rr = cache['Hier-dw=0.5'].retrieve(y_recv, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_recv, rr, kb,
                        y_original=y_orig, channel=ch, phase_align=need_align)
                elif mname == 'Stats-Only':
                    h = HierarchicalRetriever(kb, coarse_k=n_templates, fine_k=3, normalize=False,
                        distance_window=3.0, angle_window=90.0,
                        stat_filter=True, stat_max_keep=200)
                    rr = h.retrieve(y_recv, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_recv, rr, kb,
                        y_original=y_orig, channel=ch, phase_align=need_align)
                raw[mname][snr_db].append(mse(y_orig, y_recon))

    # Aggregate with trimmed mean
    results = {}
    for mname in names:
        results[mname] = [trim_mean(raw[mname][s], 0.05) for s in snr_vals]

    # Print key values
    mid = len(snr_vals) // 2
    print(f"  SNR=-150dB:")
    for mname in names:
        print(f"    {mname:20s}: {results[mname][mid]:.4f}")

    return snr_vals, results


def plot_combined(signal_data, save_path='results/exp09_template_value.png'):
    """三信号类型对比 (slow_varying, periodic, transient) 2x2 布局."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {
        'No Retrieval': '#d32f2f',
        'Physics-Only': '#1976d2',
        'Hier-dw=0.5': '#9c27b0',
        'Stats-Only': '#e65100',
        'Signal-Only': '#2e7d32',
    }
    markers = {
        'No Retrieval': 's', 'Physics-Only': '^',
        'Hier-dw=0.5': '<', 'Stats-Only': 'o', 'Signal-Only': 'v',
    }
    labels = {
        'No Retrieval': 'No Retrieval',
        'Physics-Only': 'Physics-Only',
        'Hier-dw=0.5': 'Hier-dw=0.5',
        'Stats-Only': 'Stats-Only (Ours)',
        'Signal-Only': 'Signal-Only',
    }

    for idx, (signal_type, (snr_vals, results)) in enumerate(signal_data.items()):
        ax = axes[idx]

        # Order: worst first
        order = sorted(results.keys(), key=lambda k: results[k][-1], reverse=True)

        for mname in order:
            ax.semilogy(snr_vals, results[mname],
                       color=colors[mname], marker=markers[mname],
                       markersize=5, linewidth=2 if 'Stats' in mname else 1.2,
                       alpha=0.9, label=labels[mname])

        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('MSE (log scale)', fontsize=11) if idx == 0 else None
        ax.set_title(signal_type, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.25, which='both')
        if idx == 2:
            ax.legend(fontsize=8, framealpha=0.9, loc='upper left',
                     bbox_to_anchor=(1.02, 1))

        # Annotate improvement
        no_ret = results['No Retrieval']
        stats = results['Stats-Only']
        mid = len(snr_vals) // 2
        impr = no_ret[mid] / max(stats[mid], 1e-10)
        ax.text(0.98, 0.05, f'{impr:.0f}x better\nthan direct',
               transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=2000)
    p.add_argument('--test-samples', type=int, default=50)
    p.add_argument('--snr-steps', type=int, default=9)
    args = p.parse_args()

    signal_types = ['slow_varying', 'periodic', 'transient']
    all_data = {}

    for st in signal_types:
        snr_vals, results = run_single_signal(
            st, n_templates=args.templates,
            n_test=args.test_samples, snr_steps=args.snr_steps)
        all_data[st] = (snr_vals, results)

    plot_combined(all_data)
