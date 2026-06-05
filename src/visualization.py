"""论文图表生成."""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_snr_vs_mse(
    snr_values: list,
    mse_results: dict,
    save_path: str = 'results/snr_vs_mse.png',
    title: str = 'SNR vs MSE: Retrieval Method Comparison',
):
    """核心图：SNR扫描下各检索方法的MSE对比.

    Args:
        snr_values: SNR列表 (dB)
        mse_results: {方法名: MSE列表}
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'No Retrieval': '#d32f2f',
        'Random': '#f57c00',
        'Physics-Only': '#1976d2',
        'Signal-Only': '#388e3c',
        'Hierarchical (Ours)': '#7b1fa2',
    }

    for method_name, mse_list in mse_results.items():
        color = colors.get(method_name, '#616161')
        linestyle = '-' if 'Ours' in method_name else '--'
        linewidth = 2.5 if 'Ours' in method_name else 1.5
        ax.semilogy(snr_values, mse_list, label=method_name,
                    color=color, linestyle=linestyle, linewidth=linewidth,
                    marker='o', markersize=4)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


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
