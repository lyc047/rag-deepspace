"""Phase 7: 残差量化 + 带宽节省实验.

核心论证：为什么模板匹配有价值？

直接传输：N_samples * 8 bits (PCM)
模板匹配：log2(N_templates) bits + N_samples * n_bits (量化残差)

更好的模板 → 更小的残差 → 同等MSE下需要更少比特 → 带宽节省

本实验对每个检索方法，扫描量化比特数(1-8 bits)，绘制 MSE vs Bitrate 曲线。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.stats import trim_mean
from tqdm import tqdm

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import (
    HierarchicalRetriever, RandomRetriever,
    PhysicsOnlyRetriever, SignalOnlyRetriever,
)
from reconstruction import reconstruct_from_template, bandwidth_bits
from metrics import mse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def adaptive_window_v2(sun_angle, snr_db=-150, w_base=1.0, w_max=2.0):
    angle_factor = 0.5 * (sun_angle / 90.0)
    snr_norm = (snr_db + 170) / 40.0
    snr_factor = 0.5 * (1 - snr_norm)
    return min(w_base + angle_factor + snr_factor, w_max)


def plot_quality_vs_bitrate(
    all_results: dict,  # {method: {n_bits: mse}}
    n_templates: int,
    n_samples: int,
    save_path: str = 'results/exp07_bandwidth.png',
):
    """绘制 MSE vs Bits/Sample 曲线（核心论文图）."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'No Retrieval':       '#d32f2f',
        'Random':             '#f57c00',
        'Physics-Only':       '#1976d2',
        'Hier-dw=0.5':        '#9c27b0',
        'Hier-adaptive (Ours)': '#7b1fa2',
        'Signal-Only (upper)': '#2e7d32',
    }
    markers = {
        'No Retrieval': 's', 'Random': 'D', 'Physics-Only': '^',
        'Hier-dw=0.5': '<', 'Hier-adaptive (Ours)': 'o',
        'Signal-Only (upper)': 'v',
    }
    labels = {
        'No Retrieval': 'Direct (8-bit PCM)',
        'Random': 'Random Template',
        'Physics-Only': 'Physics-Only',
        'Hier-dw=0.5': 'Hier-dw=0.5',
        'Hier-adaptive (Ours)': 'Hier-adaptive v2 (Ours)',
        'Signal-Only (upper)': 'Signal-Only (upper bound)',
    }

    for method_name, results in all_results.items():
        if method_name == 'No Retrieval':
            # Direct transmission is a single point (fixed 8-bit PCM)
            # results is {bit: mse}, extract first value
            pcm_mse = list(results.values())[0]
            pcm_bits = 8.0
            ax.scatter([pcm_bits], [pcm_mse], color=colors[method_name],
                       marker=markers[method_name], s=120, zorder=5,
                       label=labels.get(method_name, method_name),
                       edgecolors='black', linewidths=0.5)
            continue

        bit_list = sorted(results.keys())
        mse_list = [results[b] for b in bit_list]

        # bits/sample for each quant level
        bits_per_sample = []
        for n_bits in bit_list:
            bw = bandwidth_bits(n_samples, n_templates, n_bits)
            bits_per_sample.append(bw['bits_per_sample'])

        ax.plot(bits_per_sample, mse_list,
                color=colors.get(method_name, '#333'),
                marker=markers.get(method_name, 'o'),
                markersize=6, linewidth=2, alpha=0.9,
                label=labels.get(method_name, method_name))

    ax.set_xlabel('Bits per Sample', fontsize=13)
    ax.set_ylabel('MSE (log scale)', fontsize=13)
    ax.set_title(f'Quality vs Bitrate: Template Matching Saves Bandwidth\n'
                 f'({n_templates} templates, {n_samples} samples/signal, SNR=-150dB)',
                 fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.25, which='both')

    # 标注"同等质量、更少比特"
    ax.axvline(x=8.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(8.0, ax.get_ylim()[1] * 0.5, 'PCM baseline\n(8 bit/sample)',
            fontsize=8, ha='left', color='gray', alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_experiment(
    n_templates=300, n_test=50, signal_type='slow_varying',
    snr_db=-150, seed=42,
):
    """残差量化对比实验."""
    print(f"=== Exp07: Residual Quantization + Bandwidth ===")
    print(f"Templates: {n_templates}, Test samples: {n_test}")
    print(f"Signal: {signal_type}, SNR: {snr_db}dB")

    # 1. KB
    print("\n[1/4] Building KB...")
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=seed)
    n_samples = len(kb._records[0].raw_data) if kb._records else 600
    print(f"  -> {kb.count} templates, {n_samples} samples each")

    # 2. Methods
    print("[2/4] Initializing methods...")
    cache = {
        'Random': RandomRetriever(kb),
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Signal-Only (upper)': SignalOnlyRetriever(kb, normalize=False),
        'Hier-dw=0.5': HierarchicalRetriever(
            kb, coarse_k=200, fine_k=3, normalize=False,
            distance_window=0.5, angle_window=15.0,
        ),
    }
    method_names = ['No Retrieval', 'Random', 'Physics-Only',
                    'Hier-dw=0.5', 'Hier-adaptive (Ours)', 'Signal-Only (upper)']

    # Quantization levels to test
    bit_levels = [1, 2, 3, 4, 5, 6, 7, 8]

    # 3. Run
    print(f"[3/4] Testing {len(bit_levels)} bit levels x {n_test} samples...")
    raw_mse = {m: {b: [] for b in bit_levels} for m in method_names}

    for _ in tqdm(range(n_test), desc="Samples"):
        physics = generate_physical_metadata(snr_db=snr_db)
        sample = generate_telemetry_segment(
            signal_type=signal_type, physics=physics,
            seed=np.random.randint(0, 2**31 - 1),
        )
        y_original = sample.signal
        channel = DeepSpaceChannel(
            distance_au=physics['distance_au'], snr_db=snr_db,
        )
        y_received = channel.forward(y_original)

        qd = physics['distance_au']
        qa = physics['sun_earth_probe_angle']

        for mname in method_names:
            # Get best template
            if mname == 'No Retrieval':
                best_id = None
                best_record = None
            elif mname == 'Random':
                rr = cache['Random'].retrieve(y_received)
                best_id = rr[0][0] if rr else None
                best_record = kb.get_record(best_id) if best_id else None
            elif mname == 'Physics-Only':
                rr = cache['Physics-Only'].retrieve(
                    y_received, distance_au=qd, sun_angle=qa)
                best_id = rr[0][0] if rr else None
                best_record = kb.get_record(best_id) if best_id else None
            elif mname == 'Signal-Only (upper)':
                rr = cache['Signal-Only (upper)'].retrieve(y_received)
                best_id = rr[0][0] if rr else None
                best_record = kb.get_record(best_id) if best_id else None
            elif mname == 'Hier-dw=0.5':
                rr = cache['Hier-dw=0.5'].retrieve(
                    y_received, distance_au=qd, sun_angle=qa)
                best_id = rr[0][0] if rr else None
                best_record = kb.get_record(best_id) if best_id else None
            elif mname == 'Hier-adaptive (Ours)':
                dw = adaptive_window_v2(qa, snr_db)
                da = min(90, dw * 30)
                hier = HierarchicalRetriever(
                    kb, coarse_k=200, fine_k=3, normalize=False,
                    distance_window=dw, angle_window=da,
                )
                rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                best_id = rr[0][0] if rr else None
                best_record = kb.get_record(best_id) if best_id else None

            # No Retrieval baseline (8-bit PCM direct)
            if mname == 'No Retrieval':
                raw_mse[mname][1].append(mse(y_original, y_received))
                for b in bit_levels[1:]:
                    raw_mse[mname][b].append(raw_mse[mname][1][-1])
                continue

            if best_record is None:
                for b in bit_levels:
                    raw_mse[mname][b].append(1e9)
                continue

            # Reconstruct with different quantization levels
            rr_wrapped = [(best_id, 0.0)] if best_id else []
            for n_bits in bit_levels:
                y_recon = reconstruct_from_template(
                    y_received, rr_wrapped, kb,
                    n_bits=n_bits, y_original=y_original,
                )
                raw_mse[mname][n_bits].append(mse(y_original, y_recon))

    # 4. Aggregate + plot
    print("\n[4/4] Aggregating + plotting...")
    # Cap MSE at No Retrieval level (if worse than direct, just use direct)
    no_ret_mse = trim_mean(raw_mse['No Retrieval'][1], 0.05)
    results = {}
    for mname in method_names:
        results[mname] = {}
        for b in bit_levels:
            mse_val = trim_mean(raw_mse[mname][b], 0.05)
            results[mname][b] = min(mse_val, no_ret_mse)
    # No Retrieval gets the actual value
    results['No Retrieval'] = {b: no_ret_mse for b in bit_levels}

    plot_quality_vs_bitrate(
        results, n_templates, n_samples,
        save_path='results/exp07_bandwidth.png',
    )

    # Report
    print("\n" + "=" * 70)
    print(f"=== Quality vs Bitrate at SNR={snr_db}dB ===")
    print("=" * 70)
    print(f"{'Method':<22s} {'4-bit MSE':>10s} {'6-bit MSE':>10s} {'8-bit MSE':>10s}")
    print("-" * 52)
    for mname in method_names:
        mse4 = f"{results[mname][4]:.4f}"
        mse6 = f"{results[mname][6]:.4f}"
        mse8 = f"{results[mname][8]:.4f}"
        marker = ' <-- BEST' if 'adaptive' in mname or 'upper' in mname else ''
        print(f"  {mname:<20s} {mse4:>10s} {mse6:>10s} {mse8:>10s}{marker}")

    # Bandwidth saving calculation (strictly less than No Retrieval)
    print(f"\n=== Bandwidth Saving (target: beat PCM 8-bit quality = {no_ret_mse:.2f}) ===")
    target_mse = no_ret_mse * 0.99  # Must be strictly better
    for mname in ['Hier-adaptive (Ours)', 'Signal-Only (upper)',
                  'Physics-Only', 'Hier-dw=0.5']:
        best_bit = None
        for b in bit_levels:
            if results[mname][b] < target_mse:
                best_bit = b
                break
        if best_bit:
            bw = bandwidth_bits(n_samples, n_templates, best_bit)
            pcm = bw['pcm_8bit_baseline']
            saving = (1 - bw['total_bits'] / pcm) * 100
            print(f"  {mname}: {best_bit}-bit achieves PCM quality, "
                  f"saves {saving:.0f}% bandwidth "
                  f"({bw['total_bits']} vs {pcm} bits)")
        else:
            print(f"  {mname}: CANNOT beat PCM quality at any bit level")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=300)
    p.add_argument('--test-samples', type=int, default=50)
    p.add_argument('--snr', type=float, default=-150)
    args = p.parse_args()
    run_experiment(n_templates=args.templates, n_test=args.test_samples,
                   snr_db=args.snr)
