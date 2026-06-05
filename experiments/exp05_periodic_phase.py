"""Phase 5: 相位对齐 + DTW 修复周期性信号.

对比6种方法验证相位偏移问题的解决方案。

⚠️ 性能说明：DTW O(N²)计算密集型。
- 使用duration_sec=20 (200样本 vs 默认600) 加速9倍
- DTW使用Sakoe-Chiba窗口=10%信号长度

预期：Phase-AE (相位对齐欧氏距离) 和 DTW 将MSE从~1.3降至<0.1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
from scipy.stats import trim_mean
from tqdm import tqdm

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import HierarchicalRetriever, SignalOnlyRetriever
from reconstruction import reconstruct_from_template
from metrics import mse
from visualization import plot_snr_vs_mse


def adaptive_window_v2(sun_angle: float, snr_db: float = -150,
                       w_base: float = 1.0, w_max: float = 2.0) -> float:
    """自适应距离窗口 v2：太阳角 + SNR联合调节."""
    angle_factor = 0.5 * (sun_angle / 90.0)
    snr_norm = (snr_db + 170) / 40.0
    snr_factor = 0.5 * (1 - snr_norm)
    return min(w_base + angle_factor + snr_factor, w_max)


def _reconstruct_with_aligned(y_received, rr, retriever, kb):
    """使用对齐后的模板重建（若可用）."""
    best_id = rr[0][0] if rr else None
    if best_id and hasattr(retriever, 'fine'):
        aligned = retriever.fine.last_aligned_templates.get(best_id)
        if aligned is not None:
            record = kb.get_record(best_id)
            if record:
                return y_received - record.raw_data + aligned
    return reconstruct_from_template(y_received, rr, kb)


def run_experiment(
    n_templates: int = 200,
    n_test_samples: int = 30,
    snr_range: tuple = (-170, -130),
    n_snr_steps: int = 7,
    duration_sec: float = 20.0,
):
    """相位对齐 + DTW 对比实验."""
    signal_type = 'periodic'
    fs_hz = 10.0
    n_samples = int(duration_sec * fs_hz)  # 200 samples

    print(f"=== Exp05: Phase Alignment + DTW for Periodic Signals ===")
    print(f"Templates: {n_templates}, Test/SNR: {n_test_samples}")
    print(f"Signal: {signal_type}, Duration: {duration_sec}s ({n_samples} samples)")
    print(f"SNR steps: {n_snr_steps}, DTW window: {n_samples//10}")

    # 1. 知识库（使用短信号加速DTW）
    print("\n[1/4] Building knowledge base...")
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(
        n_templates=n_templates, signal_types=[signal_type],
        duration_sec=duration_sec, fs_hz=fs_hz,
    ).build(kb, seed=42)
    print(f"  -> {kb.count} templates ({n_samples} samples each)")

    # 2. 方法定义（6种）
    print("[2/4] Initializing 6 methods...")
    methods_meta = {
        'No Retrieval':              {'type': 'none'},
        'Signal-Only (Euclidean)':   {'type': 'signal', 'metric': 'euclidean'},
        'Signal-Only (Phase-AE)':    {'type': 'signal', 'metric': 'euclidean', 'phase_align': True},
        'Signal-Only (DTW)':         {'type': 'signal', 'metric': 'dtw'},
        'Hier-adapt-v2 (Euclidean)': {'type': 'hier', 'metric': 'euclidean'},
        'Hier-adapt-v2 (Phase-AE)':  {'type': 'hier', 'metric': 'euclidean', 'phase_align': True},
    }

    # 3. SNR扫描
    snr_values = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
    raw_mse = {name: {snr: [] for snr in snr_values} for name in methods_meta.keys()}
    timings = {name: 0.0 for name in methods_meta.keys()}

    total_iters = n_snr_steps * n_test_samples
    print(f"[3/4] Running SNR scan ({total_iters} iterations)...")
    for snr_db in tqdm(snr_values, desc="SNR scan"):
        for _ in range(n_test_samples):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type=signal_type, physics=physics,
                duration_sec=duration_sec, fs_hz=fs_hz,
                seed=np.random.randint(0, 2**31 - 1),
            )
            y_original = sample.signal
            channel = DeepSpaceChannel(
                distance_au=physics['distance_au'], snr_db=snr_db,
            )
            y_received = channel.forward(y_original)

            qd = physics['distance_au']
            qa = physics['sun_earth_probe_angle']

            for method_name, meta in methods_meta.items():
                if method_name == 'No Retrieval':
                    y_recon = y_received
                elif meta['type'] == 'signal':
                    t0 = time.perf_counter()
                    retriever = SignalOnlyRetriever(
                        kb, metric=meta['metric'], normalize=False,
                        phase_align=meta.get('phase_align', False),
                    )
                    rr = retriever.retrieve(y_received, k=1)
                    y_recon = _reconstruct_with_aligned(
                        y_received, rr, retriever, kb)
                    timings[method_name] += time.perf_counter() - t0
                elif meta['type'] == 'hier':
                    t0 = time.perf_counter()
                    dw = adaptive_window_v2(qa, physics['snr_db'])
                    da = min(90, dw * 30)
                    hier = HierarchicalRetriever(
                        kb, coarse_k=200, fine_k=3,
                        metric=meta['metric'], normalize=False,
                        distance_window=dw, angle_window=da,
                        phase_align=meta.get('phase_align', False),
                    )
                    rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = _reconstruct_with_aligned(
                        y_received, rr, hier, kb)
                    timings[method_name] += time.perf_counter() - t0

                raw_mse[method_name][snr_db].append(mse(y_original, y_recon))

    # 4. 聚合 + 画图
    print("\n[4/4] Aggregating + plotting...")
    results_plot = {}
    results_mean = {}
    for name in methods_meta.keys():
        results_plot[name] = [trim_mean(raw_mse[name][s], 0.05) for s in snr_values]
        results_mean[name] = [np.mean(raw_mse[name][s]) for s in snr_values]

    plot_methods = list(methods_meta.keys())
    plot_snr_vs_mse(
        snr_values=snr_values.tolist(),
        mse_results={k: results_plot[k] for k in plot_methods},
        save_path='results/exp05_periodic_phase.png',
        title=f'Phase Alignment + DTW: Periodic ({n_templates} templates, {n_samples} samples)',
        y_label='5%-Trimmed Mean MSE (log scale)',
    )

    # 报告
    print("\n" + "=" * 70)
    print("=== Results at SNR=-150dB (trimmed-mean | mean) ===")
    print("=" * 70)
    mid = len(snr_values) // 2
    for name in methods_meta.keys():
        tag = ''
        if 'Phase' in name: tag = ' ★Phase'
        elif 'DTW' in name: tag = ' ☆DTW'
        elif 'Euclidean' in name and 'Hier' in name: tag = ' [baseline]'
        print(f"  {name:30s}: TrimMean = {results_plot[name][mid]:.4f}  "
              f"| Mean = {results_mean[name][mid]:.4f}{tag}")

    # 增益对比
    baseline_euc = results_plot['Signal-Only (Euclidean)'][mid]
    print(f"\n=== Improvement over Signal-Only(Euclidean) (MSE={baseline_euc:.4f}) ===")
    for name in ['Signal-Only (Phase-AE)', 'Signal-Only (DTW)',
                 'Hier-adapt-v2 (Phase-AE)']:
        if name in results_plot:
            m = results_plot[name][mid]
            gain = (baseline_euc - m) / baseline_euc * 100
            print(f"  {name:30s}: {gain:+.1f}%")

    # 耗时统计
    print(f"\n=== Timing (total seconds) ===")
    for name in methods_meta.keys():
        if timings[name] > 0:
            print(f"  {name:30s}: {timings[name]:.1f}s "
                  f"({timings[name]/total_iters*1000:.1f}ms/query)")

    return results_plot


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=200)
    p.add_argument('--test-samples', type=int, default=30,
                   help='每个SNR点的测试样本数（默认30）')
    p.add_argument('--snr-steps', type=int, default=7,
                   help='SNR扫描步数（默认7）')
    p.add_argument('--duration', type=float, default=20.0,
                   help='信号时长秒数（默认20 → 200样本）')
    args = p.parse_args()
    run_experiment(n_templates=args.templates, n_test_samples=args.test_samples,
                   n_snr_steps=args.snr_steps, duration_sec=args.duration)
