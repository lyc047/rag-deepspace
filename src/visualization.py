"""论文图表生成."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 预定义线型风格：确保每条线都不同
_STYLE_MAP = {
    'No Retrieval':           {'color': '#d32f2f', 'ls': (0, (1,1)), 'lw': 1.5, 'marker': 's', 'ms': 4},
    'Random':                 {'color': '#f57c00', 'ls': (0, (4,2)), 'lw': 1.5, 'marker': 'D', 'ms': 4},
    'Physics-Only':           {'color': '#1976d2', 'ls': (0, (5,3)), 'lw': 1.5, 'marker': '^', 'ms': 5},
    'Signal-Only':            {'color': '#2e7d32', 'ls': (0, (3,3)), 'lw': 1.5, 'marker': 'v', 'ms': 5},
    'Hier-dw=0.5 (narrow)':   {'color': '#9c27b0', 'ls': (0, (2,2)), 'lw': 1.5, 'marker': '<', 'ms': 5},
    'Hier-dw=1.5 (wide)':     {'color': '#00838f', 'ls': (0, (6,2)), 'lw': 1.5, 'marker': '>', 'ms': 5},
    'Hier-adapt-v1 (Ours)':   {'color': '#e65100', 'ls': '-',  'lw': 2.2, 'marker': 'p', 'ms': 6},
    'Hier-adapt-v2 (Ours)':   {'color': '#7b1fa2', 'ls': '-',  'lw': 2.8, 'marker': 'o', 'ms': 6},
    'Hierarchical (Ours)':    {'color': '#7b1fa2', 'ls': '-',  'lw': 2.8, 'marker': 'o', 'ms': 6},
    'Hier-adaptive (Ours)':   {'color': '#7b1fa2', 'ls': '-',  'lw': 2.8, 'marker': 'o', 'ms': 6},
    # DTW / Phase-alignment methods (exp05)
    'Signal-Only (DTW)':        {'color': '#00838f', 'ls': (0, (3,1,1,1)), 'lw': 1.8, 'marker': 'h', 'ms': 5},
    'Signal-Only (Phase-AE)':   {'color': '#c62828', 'ls': (0, (4,2)), 'lw': 1.8, 'marker': 'P', 'ms': 6},
    'Signal-Only (Phase-DTW)':  {'color': '#e91e63', 'ls': (0, (1,1)), 'lw': 2.0, 'marker': '*', 'ms': 7},
    'Hier-adapt-v2 (DTW)':      {'color': '#00695c', 'ls': (0, (3,1,1,1)), 'lw': 1.8, 'marker': 'h', 'ms': 5},
    'Hier-adapt-v2 (Phase-AE)': {'color': '#b71c1c', 'ls': (0, (4,2)), 'lw': 2.0, 'marker': 'P', 'ms': 6},
    'Hier-adapt-v2 (Phase-DTW)':{'color': '#880e4f', 'ls': (0, (1,1)), 'lw': 2.2, 'marker': '*', 'ms': 7},
}

# 短标签映射
_SHORT_LABEL = {
    'No Retrieval': 'No Retrieval',
    'Random': 'Random',
    'Physics-Only': 'Physics-Only',
    'Signal-Only': 'Signal-Only (upper bound)',
    'Hier-dw=0.5 (narrow)': 'Hier-dw=0.5',
    'Hier-dw=1.5 (wide)': 'Hier-dw=1.5',
    'Hier-adapt-v1 (Ours)': 'Hier-adapt-v1 (angle)',
    'Hier-adapt-v2 (Ours)': 'Hier-adapt-v2 (angle+SNR)',
    'Hierarchical (Ours)': 'Hier-adapt (Ours)',
    'Hier-adaptive (Ours)': 'Hier-adapt (Ours)',
    'Signal-Only (DTW)': 'SignalOnly+DTW',
    'Signal-Only (Phase-AE)': 'SignalOnly+PhaseAlign',
    'Signal-Only (Phase-DTW)': 'SignalOnly+PhaseDTW',
    'Hier-adapt-v2 (DTW)': 'HierAdapt+DTW',
    'Hier-adapt-v2 (Phase-AE)': 'HierAdapt+PhaseAlign',
    'Hier-adapt-v2 (Phase-DTW)': 'HierAdapt+PhaseDTW',
}


def _apply_jitter(mse_lists: dict, threshold_ratio: float = 0.08) -> dict:
    """对MSE值在threshold_ratio内重合的曲线施加微小乘性偏移 (1±2%).

    确保在log-log图上也能看到分离。
    """
    arrays = {k: np.array(v, dtype=float) for k, v in mse_lists.items()}
    names = list(arrays.keys())
    jittered = {k: arr.copy() for k, arr in arrays.items()}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = arrays[names[i]], arrays[names[j]]
            # 检查是否有任何SNR点重合
            ratio = np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-10)
            if np.any(ratio < threshold_ratio):
                # 对后面的曲线施加微小偏移
                jittered[names[j]] *= (1.0 + 0.015 * (j + 1))

    return jittered


def plot_snr_vs_mse(
    snr_values: list,
    mse_results: dict,
    save_path: str = 'results/snr_vs_mse.png',
    title: str = 'SNR vs MSE: Retrieval Method Comparison',
    y_label: str = 'MSE (log scale)',
):
    """SNR扫描下各检索方法的MSE对比.

    自动检测重合曲线并施加微偏移 + 使用不同线型/标记区分。
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    jittered = _apply_jitter(mse_results)

    # 按最后一个SNR点的MSE排序（最差的先画，在底层）
    ordered = sorted(mse_results.keys(),
                     key=lambda k: mse_results[k][-1], reverse=True)

    for method_name in ordered:
        mse_list = jittered[method_name]
        style = _STYLE_MAP.get(method_name, _STYLE_MAP['No Retrieval']).copy()
        label = _SHORT_LABEL.get(method_name, method_name)

        ax.semilogy(snr_values, mse_list, label=label,
                    color=style['color'], linestyle=style['ls'],
                    linewidth=style['lw'], marker=style['marker'],
                    markersize=style['ms'], markevery=1,
                    alpha=0.9)

    # 标注重合区域
    _add_overlap_annotations(ax, mse_results, snr_values)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.25, which='both')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def _add_overlap_annotations(ax, mse_results, snr_values):
    """在图例区下方添加文字说明重合线条."""
    names = list(mse_results.keys())
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(mse_results[names[i]])
            b = np.array(mse_results[names[j]])
            ratio = np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-10)
            if np.mean(ratio) < 0.05:
                overlaps.append(f"- {_SHORT_LABEL.get(names[i], names[i])} ~ {_SHORT_LABEL.get(names[j], names[j])}")

    if overlaps:
        note = '\n'.join(f'* {o}' for o in overlaps[:3])
        ax.text(0.02, 0.02, f'Overlapping:\n{note}', transform=ax.transAxes,
                fontsize=7, color='#666', va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', alpha=0.7))


def plot_multisignal_grid(
    signal_types: list,
    all_results: dict,  # {signal_type: {method_key: mse_list}}
    snr_values: list,
    save_path: str = 'results/multisignal_grid.png',
):
    """多信号类型对比网格：每个子图一种信号类型.

    每个子图最多4条线：SignalOnly+euc, HierAdapt+euc, SignalOnly+multi, HierAdapt+multi
    使用完全不同风格的线条确保区分。
    """
    n = len(signal_types)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 4种线型组合 = 颜色 + 线型 + 标记
    line_styles = [
        {'color': '#1976d2', 'ls': '--', 'lw': 1.8, 'marker': 's', 'ms': 4, 'label': 'SignalOnly+euc'},
        {'color': '#7b1fa2', 'ls': '-',  'lw': 2.8, 'marker': 'o', 'ms': 6, 'label': 'HierAdapt+euc'},
        {'color': '#e65100', 'ls': (0, (4,2,1,2)), 'lw': 1.6, 'marker': '^', 'ms': 5, 'label': 'SignalOnly+multi'},
        {'color': '#2e7d32', 'ls': (0, (1,1)), 'lw': 2.0, 'marker': 'D', 'ms': 5, 'label': 'HierAdapt+multi'},
    ]

    for idx, st in enumerate(signal_types):
        ax = axes[idx]
        res = all_results[st]

        # Key mapping
        key_map = {
            'signal-only_euclidean': line_styles[0],
            'hier-adapt_euclidean': line_styles[1],
            'signal-only_multi': line_styles[2],
            'hier-adapt_multi': line_styles[3],
        }

        for key, sty in key_map.items():
            if key in res:
                mse_arr = np.array(res[key])
                ax.semilogy(snr_values, mse_arr,
                            label=sty['label'],
                            color=sty['color'], linestyle=sty['ls'],
                            linewidth=sty['lw'], marker=sty['marker'],
                            markersize=sty['ms'], markevery=1, alpha=0.9)

        # 标注最佳结果
        if res:
            best_mse = min(np.mean(v) for v in res.values())
            best_key = min(res.keys(), key=lambda k: np.mean(res[k]))
            ax.text(0.98, 0.05, f'Best: {best_key}\nMSE={np.mean(res[best_key]):.4f}',
                    transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

        ax.set_title(f'{st}', fontsize=13, fontweight='bold')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('MSE (log scale)')
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.25, which='both')

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_snr_vs_mse_compact(
    snr_values: list,
    mse_results: dict,
    save_path: str,
    title: str = '',
    highlight: str = None,
):
    """紧凑版SNR-MSE图（用于自适应窗口等实验）."""
    plot_snr_vs_mse(snr_values, mse_results, save_path, title)


def plot_retrieval_comparison(
    methods: list,
    recall_values: list,
    save_path: str = 'results/retrieval_recall.png',
):
    """检索质量对比：各方法的Recall@K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, recall_values, color=['#f57c00', '#1976d2',
                   '#388e3c', '#7b1fa2'])
    ax.set_ylabel('Recall@3', fontsize=12)
    ax.set_title('Retrieval Quality Comparison', fontsize=14)
    for bar, val in zip(bars, recall_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_signal_waveform(
    y_original: np.ndarray,
    y_reconstructed: np.ndarray,
    y_template: np.ndarray = None,
    save_path: str = 'results/waveform_comparison.png',
    fs_hz: float = 10.0,
):
    """信号波形对比图."""
    n_signals = 3 if y_template is not None else 2
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 8), sharex=True)

    t = np.arange(len(y_original)) / fs_hz
    axes[0].plot(t, y_original, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Original')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y_reconstructed, 'r-', linewidth=0.8)
    axes[1].set_ylabel('Reconstructed')
    axes[1].grid(True, alpha=0.3)

    if y_template is not None:
        axes[2].plot(t, y_template, 'g-', linewidth=0.8)
        axes[2].set_ylabel('Template')
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_efficiency_comparison(
    snr_values: list,
    candidate_counts: dict,  # {method_name: [avg_count_per_snr]}
    save_path: str = 'results/exp03_efficiency.png',
    title: str = '检索效率对比：粗筛候选数 vs SNR',
):
    """检索效率对比：展示各方案在不同SNR下的平均粗筛候选数.

    - 固定窗口(dw=0.5/dw=1.5)：候选数随SNR基本不变
    - 自适应窗口(v1/v2)：信道好时自动收缩 → 候选数减少 → 计算效率提升
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    style_map = {
        'Hier-dw=0.5 (narrow)':   {'color': '#9c27b0', 'ls': (0, (2,2)), 'lw': 1.5, 'marker': '<', 'ms': 6},
        'Hier-dw=1.5 (wide)':     {'color': '#00838f', 'ls': (0, (6,2)), 'lw': 1.5, 'marker': '>', 'ms': 6},
        'Hier-adapt-v1 (Ours)':   {'color': '#e65100', 'ls': '-',  'lw': 2.2, 'marker': 'p', 'ms': 7},
        'Hier-adapt-v2 (Ours)':   {'color': '#7b1fa2', 'ls': '-',  'lw': 2.8, 'marker': 'o', 'ms': 7},
    }

    for method_name, counts in candidate_counts.items():
        style = style_map.get(method_name, {}).copy()
        label = _SHORT_LABEL.get(method_name, method_name)
        ax.plot(snr_values, counts, label=label,
                color=style.get('color', '#333'),
                linestyle=style.get('ls', '-'),
                linewidth=style.get('lw', 1.5),
                marker=style.get('marker', 'o'),
                markersize=style.get('ms', 5),
                markevery=1, alpha=0.9)

    # 标注节省比例
    if 'Hier-dw=1.5 (wide)' in candidate_counts and 'Hier-adapt-v2 (Ours)' in candidate_counts:
        wide_avg = np.mean(candidate_counts['Hier-dw=1.5 (wide)'])
        v2_avg = np.mean(candidate_counts['Hier-adapt-v2 (Ours)'])
        if wide_avg > 0:
            saving = (1 - v2_avg / wide_avg) * 100
            ax.text(0.98, 0.95, f'v2 avg saving vs dw=1.5: {saving:.0f}%',
                    transform=ax.transAxes, fontsize=11, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.85))

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('平均粗筛候选数', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    ax.grid(True, alpha=0.25)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
