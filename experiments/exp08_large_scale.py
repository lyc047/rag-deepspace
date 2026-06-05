"""Phase 8: 大规模模板库效率实验.

论证自适应窗口随模板库增长的优势：
- dw=0.5 (窄): Recall急剧下降 → 漏掉最优模板
- dw=1.5 (宽): 候选数线性增长 → 计算浪费
- adaptive v2: 维持Recall ≈ 上界 + 候选数可控

扫描模板库大小: 500 → 1000 → 2000 → 5000 → 10000
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import (
    HierarchicalRetriever, SignalOnlyRetriever, CoarseRetriever,
)
from reconstruction import reconstruct_from_template
from metrics import mse, recall_at_k

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def adaptive_window_v2(sun_angle, snr_db=-150):
    angle_factor = 0.5 * (sun_angle / 90.0)
    snr_norm = (snr_db + 170) / 40.0
    snr_factor = 0.5 * (1 - snr_norm)
    return min(1.0 + angle_factor + snr_factor, 2.0)


def run_experiment(library_sizes=[500, 1000, 2000, 5000, 10000],
                   n_queries=100, snr_db=-150, seed=42):
    """大规模模板库扫描实验."""
    print(f"=== Exp08: Large-Scale Template Library ===")
    print(f"Library sizes: {library_sizes}")
    print(f"Queries per size: {n_queries}")

    np.random.seed(seed)

    # Metrics to track
    results = {
        'size': [],
        # Recall@1
        'recall_narrow': [],
        'recall_wide': [],
        'recall_adaptive': [],
        # Avg coarse candidates
        'cand_narrow': [],
        'cand_wide': [],
        'cand_adaptive': [],
        # MSE
        'mse_narrow': [],
        'mse_wide': [],
        'mse_adaptive': [],
        'mse_signal_only': [],
    }

    for n_templates in library_sizes:
        print(f"\n{'='*60}")
        print(f"Library size: {n_templates}")
        print(f"{'='*60}")

        # 1. Build KB
        kb = KnowledgeBase()
        KnowledgeBaseBuilder(
            n_templates=n_templates, signal_types=['slow_varying']
        ).build(kb, seed=seed)
        print(f"  KB: {kb.count} templates")

        # 2. Oracle (SignalOnly)
        oracle = SignalOnlyRetriever(kb, normalize=False)

        # 3. Run queries
        rec_n = []; rec_w = []; rec_a = []
        cand_n = []; cand_w = []; cand_a = []
        mse_n = []; mse_w = []; mse_a = []; mse_s = []

        for _ in tqdm(range(n_queries), desc=f"  n={n_templates}"):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type='slow_varying', physics=physics,
                seed=np.random.randint(0, 2**31 - 1),
            )
            y_original = sample.signal
            channel = DeepSpaceChannel(
                distance_au=physics['distance_au'], snr_db=snr_db,
            )
            y_received = channel.forward(y_original)

            qd = physics['distance_au']
            qa = physics['sun_earth_probe_angle']

            # Oracle
            oracle_rr = oracle.retrieve(y_received, k=1)
            oracle_id = oracle_rr[0][0] if oracle_rr else None
            mse_s.append(
                mse(y_original, reconstruct_from_template(y_received, oracle_rr, kb))
            )

            # Narrow window
            hier_n = HierarchicalRetriever(
                kb, coarse_k=500, fine_k=3, normalize=False,
                distance_window=0.5, angle_window=15.0,
            )
            rr_n = hier_n.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_n.append(1.0 if rr_n and rr_n[0][0] == oracle_id else 0.0)
            cand_n.append(hier_n.last_coarse_count)
            mse_n.append(
                mse(y_original, reconstruct_from_template(y_received, rr_n, kb))
            )

            # Wide window
            hier_w = HierarchicalRetriever(
                kb, coarse_k=500, fine_k=3, normalize=False,
                distance_window=1.5, angle_window=45.0,
            )
            rr_w = hier_w.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_w.append(1.0 if rr_w and rr_w[0][0] == oracle_id else 0.0)
            cand_w.append(hier_w.last_coarse_count)
            mse_w.append(
                mse(y_original, reconstruct_from_template(y_received, rr_w, kb))
            )

            # Adaptive
            dw = adaptive_window_v2(qa, snr_db)
            da = min(90, dw * 30)
            hier_a = HierarchicalRetriever(
                kb, coarse_k=500, fine_k=3, normalize=False,
                distance_window=dw, angle_window=da,
            )
            rr_a = hier_a.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_a.append(1.0 if rr_a and rr_a[0][0] == oracle_id else 0.0)
            cand_a.append(hier_a.last_coarse_count)
            mse_a.append(
                mse(y_original, reconstruct_from_template(y_received, rr_a, kb))
            )

        # Aggregate
        results['size'].append(n_templates)
        results['recall_narrow'].append(np.mean(rec_n))
        results['recall_wide'].append(np.mean(rec_w))
        results['recall_adaptive'].append(np.mean(rec_a))
        results['cand_narrow'].append(np.mean(cand_n))
        results['cand_wide'].append(np.mean(cand_w))
        results['cand_adaptive'].append(np.mean(cand_a))
        results['mse_narrow'].append(np.median(mse_n))
        results['mse_wide'].append(np.median(mse_w))
        results['mse_adaptive'].append(np.median(mse_a))
        results['mse_signal_only'].append(np.median(mse_s))

    # Save intermediate results
    import json
    os.makedirs('results', exist_ok=True)
    np.save('results/exp08_data.npy', results, allow_pickle=True)
    print("Data saved to results/exp08_data.npy")

    # Plot
    plot_results(results)
    return results


def plot_results(r):
    """2x2 grid: Recall, Candidates, MSE, Efficiency Ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    sizes = r['size']

    # Top-left: Recall@1
    ax = axes[0, 0]
    ax.plot(sizes, r['recall_adaptive'], 'o-', color='#7b1fa2', lw=2.5, ms=8,
            label='Hier-adaptive v2 (Ours)')
    ax.plot(sizes, r['recall_wide'], 's--', color='#00838f', lw=1.5, ms=6,
            label='Hier-dw=1.5 (wide)')
    ax.plot(sizes, r['recall_narrow'], '^:', color='#9c27b0', lw=1.5, ms=6,
            label='Hier-dw=0.5 (narrow)')
    ax.set_xlabel('Template Library Size', fontsize=12)
    ax.set_ylabel('Recall@1 (Oracle Match Rate)', fontsize=12)
    ax.set_title('Retrieval Accuracy vs Library Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Top-right: Avg Coarse Candidates
    ax = axes[0, 1]
    ax.plot(sizes, r['cand_wide'], 's--', color='#00838f', lw=1.5, ms=6,
            label='Hier-dw=1.5 (wide)')
    ax.plot(sizes, r['cand_adaptive'], 'o-', color='#7b1fa2', lw=2.5, ms=8,
            label='Hier-adaptive v2 (Ours)')
    ax.plot(sizes, r['cand_narrow'], '^:', color='#9c27b0', lw=1.5, ms=6,
            label='Hier-dw=0.5 (narrow)')
    ax.set_xlabel('Template Library Size', fontsize=12)
    ax.set_ylabel('Avg Coarse Candidates', fontsize=12)
    ax.set_title('Computational Cost vs Library Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: MSE (median)
    ax = axes[1, 0]
    ax.semilogy(sizes, r['mse_narrow'], '^:', color='#9c27b0', lw=1.5, ms=6,
                label='Hier-dw=0.5 (narrow)')
    ax.semilogy(sizes, r['mse_wide'], 's--', color='#00838f', lw=1.5, ms=6,
                label='Hier-dw=1.5 (wide)')
    ax.semilogy(sizes, r['mse_adaptive'], 'o-', color='#7b1fa2', lw=2.5, ms=8,
                label='Hier-adaptive v2 (Ours)')
    ax.semilogy(sizes, r['mse_signal_only'], 'v-', color='#2e7d32', lw=1.5, ms=5,
                label='Signal-Only (upper bound)')
    ax.set_xlabel('Template Library Size', fontsize=12)
    ax.set_ylabel('Median MSE (log scale)', fontsize=12)
    ax.set_title('Reconstruction Quality vs Library Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Bottom-right: Efficiency Ratio (recall / candidates)
    ax = axes[1, 1]
    for rec_list, cand_list, name, color, marker, ls in [
        (r['recall_adaptive'], r['cand_adaptive'], 'Adaptive v2', '#7b1fa2', 'o', '-'),
        (r['recall_wide'], r['cand_wide'], 'Wide (dw=1.5)', '#00838f', 's', '--'),
        (r['recall_narrow'], r['cand_narrow'], 'Narrow (dw=0.5)', '#9c27b0', '^', ':'),
    ]:
        eff = [rec / max(c, 1) for rec, c in zip(rec_list, cand_list)]
        ax.plot(sizes, eff, marker=marker, linestyle=ls, color=color, lw=2, ms=7,
                label=name)

    ax.set_xlabel('Template Library Size', fontsize=12)
    ax.set_ylabel('Efficiency (Recall / Candidates)', fontsize=12)
    ax.set_title('Retrieval Efficiency: Recall per Candidate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/exp08_large_scale.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: results/exp08_large_scale.png")

    # Report
    print(f"\n{'Size':<8s} {'Recall-narrow':>14s} {'Recall-adaptive':>16s} "
          f"{'Cand-wide':>10s} {'Cand-adaptive':>14s}")
    print("-" * 65)
    for i, s in enumerate(sizes):
        print(f"  {s:<6d} {r['recall_narrow'][i]:>14.3f} {r['recall_adaptive'][i]:>16.3f} "
              f"{r['cand_wide'][i]:>10.0f} {r['cand_adaptive'][i]:>14.0f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+',
                   default=[500, 1000, 2000, 5000, 10000])
    p.add_argument('--queries', type=int, default=100)
    args = p.parse_args()
    run_experiment(library_sizes=args.sizes, n_queries=args.queries)
