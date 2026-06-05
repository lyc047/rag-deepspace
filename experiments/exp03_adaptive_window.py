"""Phase 3: 自适应窗口实验.

对比7种方案 × SNR扫描：
- No Retrieval, Random, Physics-Only, Signal-Only (4基线)
- Hierarchical dw=0.5 (窄窗口)
- Hierarchical dw=1.5 (宽窗口)
- Hierarchical adaptive (自适应窗口 — 最终方案)

自适应策略：太阳角增大 → 物理参数与信号形态解耦 → 窗口线性放宽
  dw = w_min + (w_max - w_min) * (sun_angle / 90)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm
from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import (
    HierarchicalRetriever, RandomRetriever,
    PhysicsOnlyRetriever, SignalOnlyRetriever,
)
from reconstruction import reconstruct_from_template
from metrics import mse
from visualization import plot_snr_vs_mse


def adaptive_window_v1(sun_angle: float, w_min: float = 0.5, w_max: float = 1.5) -> float:
    """自适应距离窗口 v1：线性映射（保守）."""
    return w_min + (w_max - w_min) * (sun_angle / 90.0)


def adaptive_window_v2(sun_angle: float, snr_db: float = -150,
                       w_base: float = 1.0, w_max: float = 2.0) -> float:
    """自适应距离窗口 v2：太阳角 + SNR联合调节.

    物理直觉：
    - 太阳角大 → 信号形态与物理参数解耦 → 放宽窗口
    - SNR低 → 信号退化严重 → 更难匹配 → 放宽窗口
    """
    # 太阳角贡献：0 → +0.5AU
    angle_factor = 0.5 * (sun_angle / 90.0)
    # SNR贡献：-130dB(好) → 0, -170dB(差) → +0.5AU
    snr_norm = (snr_db + 170) / 40.0  # [-170,-130] → [0,1]
    snr_factor = 0.5 * snr_norm
    dw = w_base + angle_factor + snr_factor
    return min(dw, w_max)


def run_experiment(
    n_templates: int = 500,
    n_test_samples: int = 50,
    snr_range: tuple = (-170, -130),
    n_snr_steps: int = 9,
    signal_type: str = 'slow_varying',
):
    """自适应窗口对比实验."""
    print(f"=== Exp03: Adaptive Window Comparison ===")
    print(f"Templates: {n_templates}, Test/SNR: {n_test_samples}")

    # 1. 知识库
    print("[1/4] Building knowledge base...")
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type]).build(kb, seed=42)
    print(f"  -> {kb.count} templates")

    # 2. 7种方案
    print("[2/4] Initializing 7 methods...")
    methods = {
        'No Retrieval': None,
        'Random': RandomRetriever(kb),
        'Physics-Only': PhysicsOnlyRetriever(kb),
        'Signal-Only': SignalOnlyRetriever(kb, normalize=False),
        'Hier-dw=0.5 (narrow)': HierarchicalRetriever(
            kb, coarse_k=200, fine_k=3, normalize=False,
            distance_window=0.5, angle_window=15.0,
        ),
        'Hier-dw=1.5 (wide)': HierarchicalRetriever(
            kb, coarse_k=200, fine_k=3, normalize=False,
            distance_window=1.5, angle_window=45.0,
        ),
        'Hier-adapt-v1 (Ours)': 'adaptive_v1',
        'Hier-adapt-v2 (Ours)': 'adaptive_v2',
    }

    # 3. SNR扫描
    snr_values = np.linspace(snr_range[0], snr_range[1], n_snr_steps)
    results = {name: [] for name in methods.keys()}

    print(f"[3/4] Running SNR scan...")
    for snr_db in tqdm(snr_values, desc="SNR scan"):
        mse_per_method = {name: [] for name in methods.keys()}

        for _ in range(n_test_samples):
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

            for method_name, retriever in methods.items():
                if method_name == 'No Retrieval':
                    y_recon = y_received
                elif method_name == 'Hier-adapt-v1 (Ours)':
                    dw = adaptive_window_v1(qa)
                    da = min(90, dw * 30)
                    hier = HierarchicalRetriever(
                        kb, coarse_k=200, fine_k=3, normalize=False,
                        distance_window=dw, angle_window=da,
                    )
                    rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                elif method_name == 'Hier-adapt-v2 (Ours)':
                    dw = adaptive_window_v2(qa, physics['snr_db'])
                    da = min(90, dw * 30)
                    hier = HierarchicalRetriever(
                        kb, coarse_k=200, fine_k=3, normalize=False,
                        distance_window=dw, angle_window=da,
                    )
                    rr = hier.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)
                else:
                    rr = retriever.retrieve(y_received, distance_au=qd, sun_angle=qa)
                    y_recon = reconstruct_from_template(y_received, rr, kb)

                mse_per_method[method_name].append(mse(y_original, y_recon))

        for name in methods.keys():
            results[name].append(np.mean(mse_per_method[name]))

    # 4. 画图
    print("[4/4] Plotting...")
    # 选关键方法画图，避免太拥挤
    plot_methods = [
        'No Retrieval', 'Physics-Only', 'Signal-Only',
        'Hier-dw=0.5 (narrow)', 'Hier-dw=1.5 (wide)',
        'Hier-adapt-v2 (Ours)',
    ]
    plot_results = {k: results[k] for k in plot_methods}
    plot_snr_vs_mse(
        snr_values=snr_values.tolist(),
        mse_results=plot_results,
        save_path='results/exp03_adaptive_window.png',
        title=f'Adaptive Window: Hierarchical Retrieval ({signal_type})',
    )

    # 打印结果
    print("\n=== Results at SNR=-150dB ===")
    snr_idx = len(snr_values) // 2
    for name in methods.keys():
        marker = ' ★' if 'Ours' in name else ''
        print(f"  {name:30s}: MSE = {results[name][snr_idx]:.6f}{marker}")

    # 各方案增益
    baseline = results['Hier-dw=0.5 (narrow)'][snr_idx]
    for name in ['Hier-dw=1.5 (wide)', 'Hier-adapt-v1 (Ours)', 'Hier-adapt-v2 (Ours)']:
        m = results[name][snr_idx]
        gain = (baseline - m) / baseline * 100
        print(f"  {name} gain over narrow: {gain:.1f}%")

    # 自适应窗口示例
    print("\n  Adaptive Window v2 examples:")
    for a in [0, 20, 45, 70, 90]:
        for s in [-170, -150, -130]:
            print(f"    angle={a:3}° SNR={s:4}dB → dw={adaptive_window_v2(a,s):.3f}AU")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=500)
    p.add_argument('--test-samples', type=int, default=50)
    args = p.parse_args()
    run_experiment(n_templates=args.templates, n_test_samples=args.test_samples)
