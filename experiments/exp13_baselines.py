"""Phase 13: 标准压缩基线对比.

核心问题: 我们的双层压缩(模板+Rice)是否真的优于标准单层压缩?

对比基线:
  1. PCM 8-bit          (下限)
  2. DPCM + 均匀量化     (预测编码, 工程遥感最常用的轻量方案)
  3. DCT + 系数截断      (变换编码, 类似JPEG的1D版本)
  4. CCSDS 121.0 (Rice)  (当前深空标准)
  5. Ours (模板+残差+CCSDS) (语义+统计双层)

画出 Rate-Distortion 曲线, 证明语义层+统计层的叠加效果.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.stats import trim_mean
from tqdm import tqdm

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import HierarchicalRetriever
from reconstruction import reconstruct_from_template, quantize_uniform
from ccsds121 import benchmark_ccsds
from metrics import mse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Baseline 1: DPCM (Differential Pulse Code Modulation)
# ============================================================
def dpcm_encode(data: np.ndarray, n_bits: int) -> tuple:
    """一阶差分编码 + 均匀量化."""
    n = len(data)
    delta = np.zeros(n)
    reconstructed = np.zeros(n)

    for i in range(n):
        if i == 0:
            pred = 0
        else:
            pred = reconstructed[i - 1]

        delta[i] = data[i] - pred
        # Quantize delta
        dq, _ = quantize_uniform(np.array([delta[i]]), n_bits)
        reconstructed[i] = pred + dq[0]

    # Calculate bitrate: N*n_bits (we can add Rice coding later)
    bits = n * n_bits
    return reconstructed, bits


# ============================================================
# Baseline 2: DCT + coefficient truncation
# ============================================================
def dct_encode(data: np.ndarray, keep_ratio: float) -> tuple:
    """DCT变换 + 系数截断.

    Args:
        data: 输入信号
        keep_ratio: 保留的DCT系数比例 [0.01, 1.0]

    Returns:
        (reconstructed, bits_equivalent)
    """
    from scipy.fft import dct, idct

    # DCT Type II (same as JPEG's DCT)
    coeffs = dct(data, type=2, norm='ortho')
    n = len(coeffs)
    keep_n = max(1, int(n * keep_ratio))

    # Keep top keep_n coefficients by magnitude
    sorted_idx = np.argsort(np.abs(coeffs))[::-1]
    coeffs_truncated = np.zeros(n)
    for idx in sorted_idx[:keep_n]:
        coeffs_truncated[idx] = coeffs[idx]

    # Inverse DCT
    reconstructed = idct(coeffs_truncated, type=2, norm='ortho')

    # Bits: keep_n coefficients * 16 bits (position + value)
    # Position: log2(N) bits, Value: assume 8 bits
    bits_per_coeff = int(np.ceil(np.log2(n))) + 8
    bits = keep_n * bits_per_coeff

    return reconstructed, bits


# ============================================================
# Experiment
# ============================================================
def run_baseline_comparison(signal_type='slow_varying', n_templates=2000,
                            n_test=50, snr_db=-150, seed=42):
    """对一种信号类型跑所有基线的对比."""
    need_align = signal_type in ('periodic', 'transient')

    # Build KB for our method
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=seed)
    n_samples = len(kb._records[0].raw_data)

    # Bitrate levels to test
    target_bps_list = np.arange(0.5, 8.5, 0.5)

    # Store: {method: {bps: [mse_values]}}
    methods = ['PCM', 'DPCM', 'DCT', 'CCSDS121', 'Ours']
    raw = {m: [] for m in methods}  # [(bps, mse), ...] per test sample

    for _ in tqdm(range(n_test), desc=f'  {signal_type}'):
        physics = generate_physical_metadata(snr_db=snr_db)
        sample = generate_telemetry_segment(signal_type, physics=physics,
            seed=np.random.randint(0, 2**31 - 1))
        y_orig = sample.signal
        ch = DeepSpaceChannel(distance_au=physics['distance_au'], snr_db=snr_db)
        y_recv = ch.forward(y_orig)
        qd, qa = physics['distance_au'], physics['sun_earth_probe_angle']

        # ---- PCM (无信道, 纯量化失真) ----
        for b in [1, 2, 4, 8]:
            yq, step = quantize_uniform(y_orig, b)
            raw['PCM'].append((float(b), float(np.mean((y_orig - yq)**2))))

        # ---- DPCM (无信道) ----
        for b in [1, 2, 3, 4, 6, 8]:
            y_recon, bits = dpcm_encode(y_orig, b)
            # Rice编码叠加: 对重建信号做熵编码估计实际bps
            rice_r = benchmark_ccsds(y_recon)
            bps = max(0.5, min(float(b), rice_r['bits_per_sample']))
            raw['DPCM'].append((bps, float(np.mean((y_orig - y_recon)**2))))

        # ---- DCT (无信道) ----
        for ratio in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]:
            y_recon, bits = dct_encode(y_orig, ratio)
            bps = bits / n_samples
            raw['DCT'].append((bps, float(np.mean((y_orig - y_recon)**2))))

        # ---- CCSDS 121.0 (无损, bps仅统计冗余) ----
        ccsds_result = benchmark_ccsds(y_orig)
        bps = ccsds_result['bits_per_sample']
        raw['CCSDS121'].append((bps, 0.0))  # Lossless MSE=0

        # ---- Ours (template + residual, 无信道, 仅量化失真) ----
        h = HierarchicalRetriever(kb, coarse_k=n_templates, fine_k=3, normalize=False,
            distance_window=3.0, angle_window=90.0,
            stat_filter=True, stat_max_keep=200)
        rr = h.retrieve(y_recv, distance_au=qd, sun_angle=qa)
        if rr:
            info = {}
            rec = kb.get_record(rr[0][0])
            y_template = rec.raw_data if rec else np.zeros_like(y_orig)

            # Phase align
            y_template_aligned = y_template
            if need_align:
                from preprocessing import phase_align_via_cross_correlation
                aligned, lag, corr = phase_align_via_cross_correlation(y_orig, y_template)
                if corr > 0.3:
                    y_template_aligned = aligned

            residual = y_orig - y_template_aligned

            for n_bits in [1, 2, 3, 4, 6, 8]:
                rq, _ = quantize_uniform(residual, n_bits)
                y_recon = y_template_aligned + rq
                mse_val = float(np.mean((y_orig - y_recon)**2))

                # CCSDS堆叠: 残差压缩后的实际bps
                res_ccsds = benchmark_ccsds(rq)
                overhead_bps = info.get('overhead_bits', 30) / n_samples
                bps = overhead_bps + res_ccsds['bits_per_sample']
                raw['Ours'].append((bps, mse_val))

    return raw, n_samples


def plot_rate_distortion(all_data, save_path='results/exp13_baselines.png'):
    """多信号Rate-Distortion曲线对比."""
    signal_types = list(all_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {
        'PCM': '#d32f2f', 'DPCM': '#f57c00', 'DCT': '#1976d2',
        'CCSDS121': '#00838f', 'Ours': '#7b1fa2'
    }
    markers = {'PCM': 's', 'DPCM': 'D', 'DCT': '^', 'CCSDS121': 'v', 'Ours': 'o'}
    labels = {
        'PCM': 'PCM', 'DPCM': 'DPCM+Quant',
        'DCT': 'DCT', 'CCSDS121': 'CCSDS 121.0',
        'Ours': 'Ours (Template+CCSDS)'
    }

    for idx, st in enumerate(signal_types):
        ax = axes[idx]
        raw, _ = all_data[st]

        for method in ['PCM', 'DPCM', 'DCT', 'CCSDS121', 'Ours']:
            points = raw[method]
            if not points:
                continue
            # Sort by bps
            points_sorted = sorted(points, key=lambda p: p[0])
            bps_vals = [p[0] for p in points_sorted]
            mse_vals = [p[1] for p in points_sorted]

            # Bin by bps (average within 0.5 bps bins)
            bin_edges = np.arange(0, 9, 0.5)
            binned_bps = []
            binned_mse = []
            for i in range(len(bin_edges) - 1):
                mask = [(b >= bin_edges[i]) & (b < bin_edges[i+1]) for b in bps_vals]
                pts = [mse_vals[j] for j, m in enumerate(mask) if m]
                if pts:
                    binned_bps.append((bin_edges[i] + bin_edges[i+1]) / 2)
                    binned_mse.append(np.mean(pts))

            if binned_bps:
                # Handle MSE=0 (lossless) in log scale
                binned_mse_plot = [max(m, 1e-8) for m in binned_mse]
                ax.loglog(binned_bps, binned_mse_plot,
                         color=colors[method], marker=markers[method],
                         markersize=5, linewidth=2 if method == 'Ours' else 1.2,
                         alpha=0.9, label=labels[method])

        ax.set_xlabel('Bits per Sample', fontsize=11)
        if idx == 0:
            ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(st, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.25, which='both')
        if idx == 2:
            ax.legend(fontsize=8, loc='lower left')

        # Mark the CCSDS 121.0 point
        ccsds_pts = [(p[0], p[1]) for p in raw['CCSDS121']]
        if ccsds_pts:
            ax.axvline(x=np.mean([p[0] for p in ccsds_pts]),
                      color='#00838f', linestyle=':', alpha=0.4, linewidth=1)
            ax.text(np.mean([p[0] for p in ccsds_pts]) + 0.1,
                   ax.get_ylim()[1] * 0.5,
                   'CCSDS', fontsize=8, color='#00838f', alpha=0.7, rotation=90)

    fig.suptitle('Rate-Distortion Comparison: Traditional vs Ours',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def analyze_residual_distribution():
    """残差分布分析: 验证拉普拉斯假设."""
    print("\n=== Residual Distribution Analysis ===")
    for signal_type in ['slow_varying', 'periodic', 'transient']:
        kb = KnowledgeBase()
        KnowledgeBaseBuilder(n_templates=500, signal_types=[signal_type]).build(kb, seed=42)

        residuals = []
        for _ in range(200):
            sample = generate_telemetry_segment(signal_type,
                seed=np.random.randint(0, 2**31-1))
            y_orig = sample.signal

            # Find best template
            from retrieval import SignalOnlyRetriever
            so = SignalOnlyRetriever(kb, normalize=False)
            rr = so.retrieve(y_orig, k=1)  # Use clean query for oracle
            if rr:
                rec = kb.get_record(rr[0][0])
                if rec:
                    r = y_orig - rec.raw_data
                    residuals.extend(r.tolist())

        residuals = np.array(residuals)
        # Fit Laplacian: f(x) = 1/(2b)*exp(-|x-mu|/b)
        mu = np.median(residuals)
        b = np.mean(np.abs(residuals - mu))  # MLE for Laplace scale

        # Kurtosis: Laplace has excess kurtosis = 3
        from scipy import stats as sp_stats
        kurt = sp_stats.kurtosis(residuals)

        print(f"  {signal_type}:")
        print(f"    mu={mu:.6f}, b={b:.6f} (Laplace scale)")
        print(f"    kurtosis={kurt:.2f} (Laplace=3, Gaussian=0)")
        print(f"    1-bit opt? {'YES' if b < 0.1 else 'need >1 bit'}")

    return True


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=2000)
    p.add_argument('--test', type=int, default=50)
    args = p.parse_args()

    # Analyze residual distributions
    analyze_residual_distribution()

    # Run baseline comparison
    all_data = {}
    for st in ['slow_varying', 'periodic', 'transient']:
        raw, _ = run_baseline_comparison(st, n_templates=args.templates, n_test=args.test)
        all_data[st] = (raw, _)

        # Summary table
        print(f"\n  {st} summary (best achievable MSE at ~4 bps):")
        for method in ['DPCM', 'DCT', 'CCSDS121', 'Ours']:
            pts = [(p[0], p[1]) for p in raw[method] if 3.5 < p[0] < 4.5]
            if pts:
                avg_mse = np.mean([p[1] for p in pts])
                avg_bps = np.mean([p[0] for p in pts])
                print(f"    {method:10s}: bps={avg_bps:.1f}, MSE={avg_mse:.4f}")

    plot_rate_distortion(all_data)
