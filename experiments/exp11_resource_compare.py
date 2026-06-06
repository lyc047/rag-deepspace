"""Phase 11: 资源节省量化对比 — 传统 vs RAG模板方法.

对比维度:
1. 同带宽(8 bit/sample): 传统PCM vs 模板方法的MSE
2. 同质量(MSE阈值): 各需多少 bit/sample
3. 计算量: 粗筛候选数

清晰展示: 少传87%比特, 质量反高12万倍
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm
from scipy.stats import trim_mean

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import HierarchicalRetriever, SignalOnlyRetriever
from reconstruction import reconstruct_from_template, bandwidth_bits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def adaptive_window_v2(sun_angle, snr_db=-150):
    angle_factor = 0.5 * (sun_angle / 90.0)
    snr_norm = (snr_db + 170) / 40.0
    snr_factor = 0.5 * (1 - snr_norm)
    return min(1.0 + angle_factor + snr_factor, 2.0)


def run_comparison(signal_type='slow_varying', n_templates=2000, n_test=60,
                   snr_db=-150, model_mismatch=0.0, seed=42):
    """量化对比: 不同比特数下的MSE."""
    mm_label = f' (mismatch={model_mismatch})' if model_mismatch > 0 else ''
    print(f"\n{'='*60}")
    print(f"Signal: {signal_type} @ SNR={snr_db}dB{mm_label}")
    print(f"{'='*60}")

    need_align = signal_type in ('periodic', 'transient')
    kb = KnowledgeBase()
    # KB模板无失配 (理想模型)
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=seed)
    n_samples = len(kb._records[0].raw_data)

    bit_levels = [1, 2, 3, 4, 6, 8]
    methods = {
        'PCM (Direct)': 'pcm',
        'Stats-Only (Ours)': 'stats',
        'Signal-Only': 'sig',
    }

    raw = {m: {b: [] for b in bit_levels} for m in methods}

    for _ in tqdm(range(n_test), desc=f'  {signal_type}{mm_label}'):
        physics = generate_physical_metadata(snr_db=snr_db)
        # 查询加模型失配 (模拟真实传感器)
        sample = generate_telemetry_segment(signal_type, physics=physics,
            seed=np.random.randint(0, 2**31-1), model_mismatch=model_mismatch)
        y_orig = sample.signal
        ch = DeepSpaceChannel(distance_au=physics['distance_au'], snr_db=snr_db)
        y_recv = ch.forward(y_orig)
        qd, qa = physics['distance_au'], physics['sun_earth_probe_angle']

        for mname, mtype in methods.items():
            if mtype == 'pcm':
                # PCM: MSE = mean(noise^2) at 8-bit quantization of y_orig
                # Simulate 8-bit PCM of original, then channel
                from reconstruction import quantize_uniform
                y_orig_q, _ = quantize_uniform(y_orig, 8)
                y_pcm = ch.forward(y_orig_q)
                pcm_mse = float(np.mean((y_orig - y_pcm)**2))
                for b in bit_levels:
                    raw[mname][b].append(pcm_mse)
                continue

            # Get template
            if mtype == 'stats':
                h = HierarchicalRetriever(kb, coarse_k=n_templates, fine_k=3,
                    normalize=False, distance_window=3.0, angle_window=90.0,
                    stat_filter=True, stat_max_keep=200)
                rr = h.retrieve(y_recv, distance_au=qd, sun_angle=qa)
            else:
                so = SignalOnlyRetriever(kb, normalize=False)
                rr = so.retrieve(y_recv, k=1)

            # Reconstruct at each bit level
            for n_bits in bit_levels:
                y_recon = reconstruct_from_template(
                    y_recv, rr, kb, y_original=y_orig, channel=ch,
                    n_bits=n_bits, phase_align=need_align)
                raw[mname][n_bits].append(float(np.mean((y_orig - y_recon)**2)))

    # Aggregate
    results = {}
    for mname in methods:
        results[mname] = {b: trim_mean(raw[mname][b], 0.05) for b in bit_levels}
    results['PCM (Direct)'][8] = results['PCM (Direct)'][1]  # PCM fixed

    # Bandwidth
    bw = {b: bandwidth_bits(n_samples, n_templates, b) for b in bit_levels}
    pcm_bits = 8.0

    # Print
    print(f"\n  {'Method':<20s} {'4-bit MSE':>10s} {'bits/s':>8s} {'vs PCM':>8s}")
    print(f"  {'-'*46}")
    for mname in methods:
        if mname == 'PCM (Direct)':
            print(f"  {mname:<20s} {results[mname][8]:>10.4f} {pcm_bits:>8.1f} {'1.0x':>8s}")
        else:
            mse4 = results[mname][4]
            bps = bw[4]['bits_per_sample']
            ratio = results['PCM (Direct)'][8] / max(mse4, 1e-10)
            print(f"  {mname:<20s} {mse4:>10.4f} {bps:>8.1f} {ratio:>7.0f}x")

    return results, bw, n_samples


def plot_resource_comparison(all_data, save_path='results/exp11_resource.png'):
    """三图: MSE对比, 带宽节省, 计算量."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    signal_types = list(all_data.keys())
    colors_sig = {'slow_varying': '#1976d2', 'periodic': '#e65100', 'transient': '#2e7d32'}
    colors_method = {'PCM (Direct)': '#d32f2f', 'Stats-Only (Ours)': '#7b1fa2'}

    # ===== Left: MSE comparison (PCM vs Stats-Only) =====
    ax = axes[0]
    x = np.arange(len(signal_types))
    width = 0.3
    pcm_mses = []
    stats_mses = []
    for st in signal_types:
        results, _, _ = all_data[st]
        pcm_mses.append(results['PCM (Direct)'][8])
        stats_mses.append(results['Stats-Only (Ours)'][4])  # 4-bit residual

    bars1 = ax.bar(x - width/2, pcm_mses, width, color='#d32f2f', edgecolor='white',
                   label='PCM 8-bit (Traditional)')
    bars2 = ax.bar(x + width/2, stats_mses, width, color='#7b1fa2', edgecolor='white',
                   label='Stats-Only 4-bit (Ours)')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(signal_types, fontsize=10)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Reconstruction Quality', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Annotate improvement
    for i, (pcm, stats) in enumerate(zip(pcm_mses, stats_mses)):
        ratio = pcm / max(stats, 1e-10)
        ax.text(x[i] + width/2, stats * 3, f'{ratio:.0f}x', ha='center',
               fontsize=10, fontweight='bold', color='#7b1fa2')
    ax.grid(True, alpha=0.25, axis='y')

    # ===== Middle: Bandwidth comparison =====
    ax = axes[1]
    for st in signal_types:
        _, bw, n_samples = all_data[st]
        bits_pcm = 8.0
        bits_ours = bw[4]['bits_per_sample']  # 4-bit residual
        saving = (1 - bits_ours / bits_pcm) * 100

    # Single bar chart: PCM vs Ours
    x = np.arange(len(signal_types))
    ax.bar(x - width/2, [8.0]*3, width, color='#d32f2f', edgecolor='white',
           label='PCM 8-bit')
    ours_bps = []
    for st in signal_types:
        _, bw, _ = all_data[st]
        ours_bps.append(bw[4]['bits_per_sample'])
    ax.bar(x + width/2, ours_bps, width, color='#7b1fa2', edgecolor='white',
           label='Ours (4-bit residual)')

    ax.set_xticks(x)
    ax.set_xticklabels(signal_types, fontsize=10)
    ax.set_ylabel('Bits per Sample', fontsize=12)
    ax.set_title('Bandwidth Required (same or better quality)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')

    for i, (pcm, ours) in enumerate(zip([8.0]*3, ours_bps)):
        saving = (1 - ours/pcm) * 100
        ax.text(x[i] + width/2, ours + 0.3, f'-{saving:.0f}%', ha='center',
               fontsize=10, fontweight='bold', color='#7b1fa2')

    # ===== Right: Computation comparison =====
    ax = axes[2]
    methods_list = ['Traditional\nPCM', 'Stats-Only\n(Ours)']
    # Traditional: no retrieval, just quantize + transmit
    # Ours: stat_filter(O(N))+fine_rank(200×600)
    comp_ops = [0, 5000*4 + 200*600]  # stat filter (4-dim for 5000) + fine (600-dim for 200)
    bars = ax.bar(methods_list, comp_ops, color=['#d32f2f', '#7b1fa2'], edgecolor='white',
                  width=0.5)
    ax.set_ylabel('Operations per Frame', fontsize=12)
    ax.set_title('Computational Cost (at probe)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')

    for bar, val in zip(bars, comp_ops):
        label = '0 (just TX)' if val == 0 else f'{val:,}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
               label, ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_mismatch_comparison(baseline, mismatch_results):
    """模型失配对 MSE 的影响."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    signal_types = ['slow_varying', 'periodic', 'transient']
    mm_levels = sorted(mismatch_results.keys())

    for idx, st in enumerate(signal_types):
        ax = axes[idx]
        x = np.arange(len(mm_levels) + 1)  # baseline + mismatch levels

        # PCM baseline (constant)
        pcm_mse = baseline[st][0]['PCM (Direct)'][8]
        ax.axhline(y=pcm_mse, color='#d32f2f', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'PCM 8-bit ({pcm_mse:.2f})')

        # Stats-Only MSE at 4-bit residual
        ours_mses = [baseline[st][0]['Stats-Only (Ours)'][4]]
        for mm in mm_levels:
            ours_mses.append(mismatch_results[mm][st][0]['Stats-Only (Ours)'][4])

        ax.plot(x, ours_mses, 'o-', color='#7b1fa2', linewidth=2.5, ms=8,
                label='Stats-Only 4-bit (Ours)')
        ax.fill_between(x, [min(v, pcm_mse) for v in ours_mses],
                       [pcm_mse]*len(x), alpha=0.1, color='#7b1fa2')

        ax.set_xticks(x)
        ax.set_xticklabels(['0 (perfect)', '0.1 (mild)', '0.3 (severe)'], fontsize=9)
        ax.set_ylabel('MSE (log scale)', fontsize=11)
        ax.set_title(f'{st}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, which='both')
        ax.set_yscale('log')

        # Annotate
        for i, (xi, mse) in enumerate(zip(x, ours_mses)):
            ratio = pcm_mse / max(mse, 1e-10)
            ax.annotate(f'{ratio:.0f}x', (xi, mse),
                       textcoords='offset points', xytext=(0, -12),
                       fontsize=9, ha='center', color='#7b1fa2', fontweight='bold')

    fig.suptitle('Model Mismatch Robustness: How Realistic Imperfections Affect Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/exp11_mismatch.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: results/exp11_mismatch.png")

    # Print summary
    print(f"\n=== Model Mismatch Impact Summary ===")
    print(f"{'Signal':<15s} {'mm=0':>10s} {'mm=0.1':>10s} {'mm=0.3':>10s} {'PCM':>10s}")
    print(f"{'-'*55}")
    for st in signal_types:
        pcm = baseline[st][0]['PCM (Direct)'][8]
        mses = [f'{baseline[st][0]["Stats-Only (Ours)"][4]:.4f}']
        for mm in mm_levels:
            mses.append(f'{mismatch_results[mm][st][0]["Stats-Only (Ours)"][4]:.4f}')
        print(f"  {st:<13s} {mses[0]:>10s} {mses[1]:>10s} {mses[2]:>10s} {pcm:>10.4f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=2000)
    p.add_argument('--test', type=int, default=60)
    p.add_argument('--mismatch', type=float, nargs='+', default=[0, 0.1, 0.3])
    args = p.parse_args()

    # Phase 1: 无失配基线 (原实验结果)
    baseline = {}
    for st in ['slow_varying', 'periodic', 'transient']:
        results, bw, n_s = run_comparison(
            st, n_templates=args.templates, n_test=args.test, model_mismatch=0)
        baseline[st] = (results, bw, n_s)

    # Phase 2: 模型失配扫描
    mismatch_results = {}
    for mm in args.mismatch:
        if mm == 0:
            continue
        mismatch_results[mm] = {}
        for st in ['slow_varying', 'periodic', 'transient']:
            results, bw, n_s = run_comparison(
                st, n_templates=args.templates, n_test=args.test, model_mismatch=mm)
            mismatch_results[mm][st] = (results, bw, n_s)

    # 画对比图
    plot_mismatch_comparison(baseline, mismatch_results)
