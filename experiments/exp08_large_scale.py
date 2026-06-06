"""Phase 8: 大规模模板库效率实验.

两栏图：
- 左: Recall@1 vs 库大小 (自适应 vs 窄窗口)
- 右: 条件分层对比 — 自适应 vs dw=1.5 在不同信道下
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


def adaptive_window_v2(sun_angle, snr_db=-150):
    angle_factor = 0.5 * (sun_angle / 90.0)
    snr_norm = (snr_db + 170) / 40.0
    snr_factor = 0.5 * (1 - snr_norm)
    return min(1.0 + angle_factor + snr_factor, 2.0)


def run_scaling_experiment(library_sizes=[500, 1000, 2000, 5000, 10000],
                           n_queries=100, snr_db=-150, seed=42):
    """Recall vs Library Size 扫描."""
    np.random.seed(seed)
    results = {'size': [], 'recall_narrow': [], 'recall_adaptive': []}

    for n_templates in library_sizes:
        kb = KnowledgeBase()
        KnowledgeBaseBuilder(n_templates=n_templates, signal_types=['slow_varying']
                           ).build(kb, seed=seed)
        oracle = SignalOnlyRetriever(kb, normalize=False)
        coarse_k = max(200, n_templates)

        rec_n, rec_a = [], []
        for _ in range(n_queries):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type='slow_varying', physics=physics,
                seed=np.random.randint(0, 2**31 - 1))
            y_original = sample.signal
            ch = DeepSpaceChannel(distance_au=physics['distance_au'], snr_db=snr_db)
            y_received = ch.forward(y_original)
            qd, qa = physics['distance_au'], physics['sun_earth_probe_angle']

            oracle_rr = oracle.retrieve(y_received, k=1)
            oracle_id = oracle_rr[0][0] if oracle_rr else None

            # Narrow
            h = HierarchicalRetriever(kb, coarse_k=coarse_k, fine_k=3, normalize=False,
                                      distance_window=0.5, angle_window=15.0)
            rr = h.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_n.append(1.0 if rr and rr[0][0] == oracle_id else 0.0)

            # Adaptive
            dw = adaptive_window_v2(qa, snr_db)
            da = min(90, dw * 30)
            h = HierarchicalRetriever(kb, coarse_k=coarse_k, fine_k=3, normalize=False,
                                      distance_window=dw, angle_window=da)
            rr = h.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_a.append(1.0 if rr and rr[0][0] == oracle_id else 0.0)

        results['size'].append(n_templates)
        results['recall_narrow'].append(np.mean(rec_n))
        results['recall_adaptive'].append(np.mean(rec_a))
        print(f"  n={n_templates}: recall narrow={np.mean(rec_n):.3f}, adaptive={np.mean(rec_a):.3f}")

    return results


def run_stratified_experiment(n_templates=5000, n_per_condition=150, seed=42):
    """条件分层: 对比不同 SNR/角度下的 Recall 和候选数."""
    np.random.seed(seed)
    kb = KnowledgeBase()
    KnowledgeBaseBuilder(n_templates=n_templates, signal_types=['slow_varying']
                       ).build(kb, seed=seed)
    oracle = SignalOnlyRetriever(kb, normalize=False)
    coarse_k = max(200, n_templates)

    conditions = [
        # (label, snr_db, sun_angle)
        ('Good\nSNR=-130\nangle=15°', -130, 15),
        ('Typical\nSNR=-150\nangle=45°', -150, 45),
        ('Bad\nSNR=-170\nangle=75°', -170, 75),
    ]

    results = {}
    for label, snr_db, sun_angle in conditions:
        rec_w = []; cand_w = []
        rec_a = []; cand_a = []
        dw_a_vals = []

        for _ in range(n_per_condition):
            physics = generate_physical_metadata(snr_db=snr_db)
            sample = generate_telemetry_segment(
                signal_type='slow_varying', physics=physics,
                seed=np.random.randint(0, 2**31 - 1))
            y_original = sample.signal
            ch = DeepSpaceChannel(distance_au=physics['distance_au'], snr_db=snr_db)
            y_received = ch.forward(y_original)
            qd = physics['distance_au']
            qa = sun_angle  # 固定太阳角

            oracle_rr = oracle.retrieve(y_received, k=1)
            oracle_id = oracle_rr[0][0] if oracle_rr else None

            # Wide (dw=1.5)
            h_w = HierarchicalRetriever(kb, coarse_k=coarse_k, fine_k=3, normalize=False,
                                        distance_window=1.5, angle_window=45.0)
            rr_w = h_w.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_w.append(1.0 if rr_w and rr_w[0][0] == oracle_id else 0.0)
            cand_w.append(h_w.last_coarse_count)

            # Adaptive
            dw = adaptive_window_v2(qa, snr_db)
            da = min(90, dw * 30)
            h_a = HierarchicalRetriever(kb, coarse_k=coarse_k, fine_k=3, normalize=False,
                                        distance_window=dw, angle_window=da)
            rr_a = h_a.retrieve(y_received, distance_au=qd, sun_angle=qa)
            rec_a.append(1.0 if rr_a and rr_a[0][0] == oracle_id else 0.0)
            cand_a.append(h_a.last_coarse_count)
            dw_a_vals.append(dw)

        results[label] = {
            'dw_adaptive': np.mean(dw_a_vals),
            'recall_wide': np.mean(rec_w),
            'recall_adaptive': np.mean(rec_a),
            'cand_wide': np.mean(cand_w),
            'cand_adaptive': np.mean(cand_a),
        }

    return results


def plot_combined(scaling, stratified):
    """双栏图: 左=Recall vs Size, 右=条件分层."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ===== LEFT: Recall@1 vs Library Size =====
    sizes = scaling['size']
    ax1.plot(sizes, scaling['recall_adaptive'], 'o-', color='#7b1fa2', lw=3, ms=10,
             label='Hier-adaptive v2 (Ours)', zorder=5)
    ax1.plot(sizes, scaling['recall_narrow'], '^:', color='#9c27b0', lw=1.8, ms=7,
             label='Hier-dw=0.5 (narrow)')

    for i in range(len(sizes)):
        ax1.fill_between([sizes[i]-200, sizes[i]+200],
                         [scaling['recall_narrow'][i]]*2,
                         [scaling['recall_adaptive'][i]]*2,
                         alpha=0.08, color='#7b1fa2')

    ax1.set_xlabel('Template Library Size', fontsize=13)
    ax1.set_ylabel('Recall@1', fontsize=13)
    ax1.set_title('Retrieval Accuracy vs Library Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(0, 1.05)

    best_idx = np.argmax(scaling['recall_adaptive'])
    ax1.annotate(f'{scaling["recall_adaptive"][best_idx]:.0%}',
                xy=(sizes[best_idx], scaling['recall_adaptive'][best_idx]),
                xytext=(sizes[best_idx], scaling['recall_adaptive'][best_idx]+0.12),
                fontsize=13, fontweight='bold', color='#7b1fa2', ha='center',
                arrowprops=dict(arrowstyle='->', color='#7b1fa2', lw=2))

    # ===== RIGHT: Stratified comparison =====
    labels = list(stratified.keys())
    x = np.arange(len(labels))
    width = 0.3

    # Recall bars
    rec_wide = [stratified[l]['recall_wide'] for l in labels]
    rec_adap = [stratified[l]['recall_adaptive'] for l in labels]
    bars1 = ax2.bar(x - width/2, rec_wide, width, color='#00838f', edgecolor='white',
                    label='dw=1.5 (fixed wide)')
    bars2 = ax2.bar(x + width/2, rec_adap, width, color='#7b1fa2', edgecolor='white',
                    label='Adaptive v2 (Ours)')

    # Annotate bars
    for bar, val in zip(bars1, rec_wide):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=10, fontweight='bold', color='#00838f')
    for bar, val in zip(bars2, rec_adap):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=10, fontweight='bold', color='#7b1fa2')

    # Candidate count as text below
    for i, label in enumerate(labels):
        s = stratified[label]
        dw_val = s['dw_adaptive']
        cand_saving = (s['cand_wide'] - s['cand_adaptive']) / max(s['cand_wide'], 1) * 100
        ax2.text(i, max(rec_wide[i], rec_adap[i]) + 0.18,
                f'dw={dw_val:.1f}AU\n'
                f'cand: {s["cand_adaptive"]:.0f} vs {s["cand_wide"]:.0f}\n'
                f'({cand_saving:+.0f}%)',
                ha='center', fontsize=8, color='#555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', alpha=0.8))

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Recall@1', fontsize=13)
    ax2.set_title(f'Performance by Channel Condition ({n_templates_strat} templates)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.25, axis='y')
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('results/exp08_large_scale.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved: results/exp08_large_scale.png")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+',
                   default=[500, 1000, 2000, 5000, 10000])
    p.add_argument('--queries', type=int, default=100)
    p.add_argument('--stratified-templates', type=int, default=5000)
    p.add_argument('--stratified-samples', type=int, default=150)
    args = p.parse_args()

    print("=== Phase 1: Recall vs Library Size ===")
    scaling = run_scaling_experiment(library_sizes=args.sizes, n_queries=args.queries)

    print(f"\n=== Phase 2: Stratified by Condition ({args.stratified_templates} templates) ===")
    n_templates_strat = args.stratified_templates
    stratified = run_stratified_experiment(
        n_templates=args.stratified_templates,
        n_per_condition=args.stratified_samples)

    for label, s in stratified.items():
        dw = s['dw_adaptive']
        print(f"  {label.replace(chr(10),' ')}: dw={dw:.2f}AU, "
              f"recall: wide={s['recall_wide']:.3f} adaptive={s['recall_adaptive']:.3f}, "
              f"cand: wide={s['cand_wide']:.0f} adaptive={s['cand_adaptive']:.0f}")

    plot_combined(scaling, stratified)
