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
    """对一种信号类型跑所有基线的对比, 每个整数bps一个点."""
    need_align = signal_type in ('periodic', 'transient')
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=seed)
    n_samples = len(kb._records[0].raw_data)

    target_bps_list = list(range(1, 9))  # 1,2,...,8 bps

    methods = ['PCM', 'DPCM', 'DCT', 'CCSDS121', 'Ours']
    # raw[method][target_bps] = [mse_values...]
    raw = {m: {b: [] for b in target_bps_list} for m in methods}

    for _ in tqdm(range(n_test), desc=f'  {signal_type}'):
        physics = generate_physical_metadata(snr_db=snr_db)
        sample = generate_telemetry_segment(signal_type, physics=physics,
            seed=np.random.randint(0, 2**31 - 1))
        y_orig = sample.signal

        # ---- PCM: bps = n_bits exactly ----
        for b in target_bps_list:
            yq, _ = quantize_uniform(y_orig, b)
            raw['PCM'][b].append(float(np.mean((y_orig - yq)**2)))

        # ---- DPCM: bps ≈ n_bits, use Rice to get real bps ----
        for b in target_bps_list:
            y_recon, bits = dpcm_encode(y_orig, b)
            rice_r = benchmark_ccsds(y_recon)
            actual_bps = max(0.5, min(float(b), rice_r['bits_per_sample']))
            # Bin to nearest integer bps
            b_target = int(round(actual_bps))
            if 1 <= b_target <= 8:
                raw['DPCM'][b_target].append(float(np.mean((y_orig - y_recon)**2)))

        # ---- DCT: sweep keep_ratio to hit each integer bps ----
        dct_points = []
        for ratio in np.linspace(0.01, 1.0, 30):
            y_recon, bits = dct_encode(y_orig, ratio)
            bps = bits / n_samples
            mse_val = float(np.mean((y_orig - y_recon)**2))
            dct_points.append((bps, mse_val))
        dct_points.sort(key=lambda p: p[0])
        # Assign each DCT point to nearest integer bps (take closest)
        for target_b in target_bps_list:
            best = min(dct_points, key=lambda p: abs(p[0] - target_b))
            raw['DCT'][target_b].append(best[1])

        # ---- CCSDS 121.0 ----
        ccsds_r = benchmark_ccsds(y_orig)
        bps = ccsds_r['bits_per_sample']
        b_target = int(round(bps))
        if 1 <= b_target <= 8:
            raw['CCSDS121'][b_target].append(0.0)

        # ---- Ours ----
        h = HierarchicalRetriever(kb, coarse_k=n_templates, fine_k=3, normalize=False,
            distance_window=3.0, angle_window=90.0,
            stat_filter=True, stat_max_keep=200)
        rr = h.retrieve(y_orig, distance_au=1.5, sun_angle=45)
        if not rr:
            continue
        rec = kb.get_record(rr[0][0])
        if not rec:
            continue
        y_template = rec.raw_data

        # Phase align
        y_template_aligned = y_template
        if need_align:
            from preprocessing import phase_align_via_cross_correlation
            aligned, lag, corr = phase_align_via_cross_correlation(y_orig, y_template)
            if corr > 0.3:
                y_template_aligned = aligned

        residual = y_orig - y_template_aligned

        # Sweep n_bits 1..8, compute actual bps from CCSDS on residual
        ours_points = []
        for n_bits in target_bps_list:
            rq, _ = quantize_uniform(residual, n_bits)
            mse_val = float(np.mean((y_orig - (y_template_aligned + rq))**2))
            res_ccsds = benchmark_ccsds(rq)
            overhead_bps = 30.0 / n_samples  # ~0.05 bps
            actual_bps = overhead_bps + res_ccsds['bits_per_sample']
            ours_points.append((actual_bps, mse_val))

        for target_b in target_bps_list:
            best = min(ours_points, key=lambda p: abs(p[0] - target_b))
            raw['Ours'][target_b].append(best[1])

    # Aggregate: mean MSE per target bps
    results = {}
    for method in methods:
        results[method] = {}
        for b in target_bps_list:
            vals = raw[method][b]
            if vals:
                results[method][b] = trim_mean(vals, 0.05)
            else:
                results[method][b] = None

    return results


def plot_rate_distortion(all_data, save_path='results/exp13_baselines.png'):
    """多信号Rate-Distortion曲线, 每整数bps一个点."""
    signal_types = list(all_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {'PCM': '#d32f2f', 'DPCM': '#f57c00', 'DCT': '#1976d2',
              'CCSDS121': '#00838f', 'Ours': '#7b1fa2'}
    markers = {'PCM': 's', 'DPCM': 'D', 'DCT': '^', 'CCSDS121': 'v', 'Ours': 'o'}
    labels = {'PCM': 'PCM', 'DPCM': 'DPCM+Quant', 'DCT': 'DCT',
              'CCSDS121': 'CCSDS 121.0', 'Ours': 'Ours (Template+CCSDS)'}

    for idx, st in enumerate(signal_types):
        ax = axes[idx]
        results = all_data[st]

        for method in ['PCM', 'DPCM', 'DCT', 'CCSDS121', 'Ours']:
            data = results[method]
            bps_list = []
            mse_list = []
            for b in sorted(data.keys()):
                if data[b] is not None:
                    bps_list.append(float(b))
                    mse_val = data[b]
                    mse_list.append(max(mse_val, 1e-8) if mse_val == 0 else mse_val)

            if bps_list:
                ax.semilogy(bps_list, mse_list,
                           color=colors[method], marker=markers[method],
                           markersize=7, linewidth=2.5 if method == 'Ours' else 1.5,
                           alpha=0.9, label=labels[method])

        ax.set_xlabel('Bits per Sample', fontsize=11)
        if idx == 0:
            ax.set_ylabel('MSE (log scale)', fontsize=11)
        ax.set_title(st, fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, 9))
        ax.grid(True, alpha=0.25, which='both')
        if idx == 2:
            ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('Rate-Distortion: Traditional Codecs vs RAG Template Method',
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
        results = run_baseline_comparison(st, n_templates=args.templates, n_test=args.test)
        all_data[st] = results

        # Summary: MSE at 4 bps
        print(f"\n  {st} @ 4 bps:")
        for method in ['PCM', 'DPCM', 'DCT', 'Ours']:
            mse_val = results[method].get(4, None)
            if mse_val is not None:
                print(f"    {method:10s}: MSE={mse_val:.6f}")

    plot_rate_distortion(all_data)
