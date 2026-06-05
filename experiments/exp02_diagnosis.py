"""Phase 2 诊断实验：系统分析粗筛失败模式.

核心问题："在什么(距离, 太阳角)条件下，物理粗筛漏掉了最优模板？"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm
from collections import defaultdict

from telemetry import generate_telemetry_segment, generate_physical_metadata
from channel import DeepSpaceChannel
from knowledge import KnowledgeBase, KnowledgeBaseBuilder
from retrieval import CoarseRetriever, SignalOnlyRetriever

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_diagnosis(n_templates=500, n_queries=300, signal_type='slow_varying'):
    """运行诊断实验."""
    print(f"=== Exp02: Coarse Filter Failure Diagnosis ===")
    print(f"Templates: {n_templates}, Queries: {n_queries}")

    # 1. 构建知识库
    print("[1/3] Building knowledge base...")
    kb = KnowledgeBase()
    builder = KnowledgeBaseBuilder(n_templates=n_templates, signal_types=[signal_type])
    builder.build(kb, seed=42)
    print(f"  -> {kb.count} templates")

    # 2. 检索器
    print("[2/3] Initializing retrievers...")
    coarse = CoarseRetriever(kb)
    oracle_searcher = SignalOnlyRetriever(kb, metric='euclidean', normalize=False)

    # 3. 诊断循环
    print(f"[3/3] Running {n_queries} queries...")
    records = []   # [{query_dist, query_angle, delta_dist, delta_angle, is_hit, ...}]
    window_stats = defaultdict(list)

    for _ in tqdm(range(n_queries), desc="Diagnosis"):
        physics = generate_physical_metadata()
        sample = generate_telemetry_segment(
            signal_type=signal_type, physics=physics,
            seed=np.random.randint(0, 2**31 - 1),
        )
        channel = DeepSpaceChannel(distance_au=physics['distance_au'],
                                   snr_db=physics['snr_db'])
        y_received = channel.forward(sample.signal)

        # Oracle: SignalOnly全库搜索
        oracle_results = oracle_searcher.retrieve(y_received, k=1)
        if not oracle_results:
            continue
        oracle_id = oracle_results[0][0]
        oracle_rec = kb.get_record(oracle_id)
        if oracle_rec is None:
            continue
        oracle_p = oracle_rec.physics

        qd = physics['distance_au']
        qa = physics['sun_earth_probe_angle']

        # 默认窗口粗筛
        candidates_default = coarse.retrieve(
            distance_au=qd, sun_angle=qa,
            k=200, distance_window=0.5, angle_window=15.0,
        )
        cand_ids = [c[0] for c in candidates_default]
        is_hit = oracle_id in cand_ids

        records.append({
            'query_dist': qd,
            'query_angle': qa,
            'delta_dist': abs(qd - oracle_p['distance_au']),
            'delta_angle': abs(qa - oracle_p['sun_earth_probe_angle']),
            'is_hit': is_hit,
        })

        # 不同窗口大小的Recall扫描
        for dw in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            cand_w = coarse.retrieve(
                distance_au=qd, sun_angle=qa,
                k=500, distance_window=dw, angle_window=min(90, dw * 30),
            )
            window_stats[f'dw={dw}'].append(
                1.0 if oracle_id in [c[0] for c in cand_w] else 0.0
            )

    hits = [r for r in records if r['is_hit']]
    misses = [r for r in records if not r['is_hit']]
    total = len(records)
    print(f"\nResults: {len(hits)} hits ({len(hits)/total*100:.1f}%), "
          f"{len(misses)} misses ({len(misses)/total*100:.1f}%)")

    return records, hits, misses, window_stats


def plot_and_report(records, hits, misses, window_stats):
    """生成4张诊断图 + 文字报告."""
    os.makedirs('results', exist_ok=True)
    total = len(records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 图1: 散点图 — 查询太阳角 vs Oracle距离差异
    ax = axes[0, 0]
    hx, hy = [r['query_angle'] for r in hits], [r['delta_dist'] for r in hits]
    mx, my = [r['query_angle'] for r in misses], [r['delta_dist'] for r in misses]
    ax.scatter(hx, hy, c='#4caf50', alpha=0.3, s=8, label=f'Hit ({len(hits)})')
    ax.scatter(mx, my, c='#f44336', alpha=0.5, s=15, label=f'Miss ({len(misses)})')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Default window (0.5AU)')
    ax.set_xlabel('Sun-Earth-Probe Angle (°)', fontsize=11)
    ax.set_ylabel('|Query Distance - Oracle Distance| (AU)', fontsize=11)
    ax.set_title('Hit vs Miss: Physics Delta Analysis', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2: 不同太阳角区间的 Recall
    ax = axes[0, 1]
    bins = [(0,15),(15,30),(30,45),(45,60),(60,75),(75,90)]
    labels = [f'{a}-{b}°' for a,b in bins]
    rates = []
    for lo, hi in bins:
        group = [r for r in records if lo <= r['query_angle'] < hi]
        rates.append(sum(1 for r in group if r['is_hit']) / len(group) if group else 0)
    colors = ['#d32f2f' if r < 0.5 else '#388e3c' for r in rates]
    ax.bar(labels, rates, color=colors)
    ax.set_ylabel('Recall (Oracle in Coarse Candidates)', fontsize=11)
    ax.set_title('Coarse Filter Recall by Sun Angle Range', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, r) in enumerate(zip(ax.patches, rates)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{r:.1%}', ha='center', fontsize=9)

    # 图3: 窗口大小 → Recall曲线
    ax = axes[1, 0]
    ws = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    wrecall = [np.mean(window_stats[f'dw={w}']) for w in ws]
    ax.plot(ws, wrecall, 'o-', color='#7b1fa2', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% target')
    ax.set_xlabel('Distance Window (AU)', fontsize=11)
    ax.set_ylabel('Recall@Coarse', fontsize=11)
    ax.set_title('Recall → Window Size: Find the Sweet Spot', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for w, r in zip(ws, wrecall):
        if r >= 0.95:
            print(f"  ✓ 95% recall at dw={w}AU")
            ax.axvline(x=w, color='green', linestyle=':', alpha=0.5)
            break

    # 图4: delta_dist 直方图
    ax = axes[1, 1]
    all_deltas = [r['delta_dist'] for r in records]
    ax.hist(all_deltas, bins=40, color='#1976d2', alpha=0.7, edgecolor='white')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Default (0.5AU)')
    pct = sum(1 for d in all_deltas if d <= 0.5) / len(all_deltas) * 100
    ax.set_xlabel('|Query Dist - Oracle Dist| (AU)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distance to Oracle: {pct:.1f}% within 0.5AU', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('results/exp02_diagnosis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: results/exp02_diagnosis.png")

    # 文字报告
    print("\n=== Diagnosis Report ===")
    print(f"  Total queries: {total}")
    print(f"  Default window (0.5AU, 15°) recall: {len(hits)/total*100:.1f}%")
    if misses:
        avg_dd = np.mean([m['delta_dist'] for m in misses])
        avg_da = np.mean([m['delta_angle'] for m in misses])
        print(f"  Avg miss: delta_dist={avg_dd:.3f}AU, delta_angle={avg_da:.1f}°")
        # 小太阳角miss率
        small = [r for r in records if r['query_angle'] < 30]
        large = [r for r in records if r['query_angle'] >= 30]
        if small:
            s_rate = sum(1 for r in small if not r['is_hit']) / len(small)
            print(f"  Miss rate sun<30°:  {s_rate*100:.1f}%")
        if large:
            l_rate = sum(1 for r in large if not r['is_hit']) / len(large)
            print(f"  Miss rate sun>=30°: {l_rate*100:.1f}%")
    print(f"  Window→Recall: {dict((k, f'{np.mean(v):.2%}') for k,v in sorted(window_stats.items()))}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=int, default=500)
    p.add_argument('--queries', type=int, default=300)
    args = p.parse_args()
    records, hits, misses, ws = run_diagnosis(
        n_templates=args.templates, n_queries=args.queries,
    )
    plot_and_report(records, hits, misses, ws)
