"""信号重建模块：模板匹配 + 残差传输 + 量化.

Phase 4 完整流水线:
1. 发射端: y_original → 找模板 → [相位对齐] → residual = y_original - y_template_aligned
2. 残差过信道 (AWGN)
3. 接收端: y_recon = y_template_aligned + received_residual

模板越好 + 对齐 → 残差功率越小 → 同等SNR下绝对噪声越小 → MSE越低
"""
import numpy as np
from typing import List, Tuple, Optional
from knowledge import KnowledgeBase
from preprocessing import phase_align_via_cross_correlation


def quantize_uniform(
    x: np.ndarray,
    n_bits: int,
    max_val: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """均匀量化器."""
    n_bits = max(1, min(n_bits, 16))
    if max_val is None:
        max_val = float(np.max(np.abs(x)))
    if max_val < 1e-12:
        return np.zeros_like(x), 0.0
    levels = 2 ** n_bits - 1
    step = 2.0 * max_val / levels
    x_clipped = np.clip(x, -max_val, max_val)
    return np.round(x_clipped / step) * step, step


def reconstruct_from_template(
    y_received: np.ndarray,
    retrieval_results: List[Tuple[int, float]],
    kb: KnowledgeBase,
    y_original: Optional[np.ndarray] = None,
    channel: object = None,
    n_bits: Optional[int] = None,
    phase_align: bool = False,
) -> np.ndarray:
    """模板+残差重建 (发射端残差过信道).

    Args:
        y_received: 接收端收到的含噪信号 (用于找模板)
        retrieval_results: [(template_id, distance), ...]
        kb: 知识库
        y_original: 原始干净信号 (用于算残差), None时退回直接传
        channel: DeepSpaceChannel实例, 用于对残差加噪. None时完美传输
        n_bits: 残差量化比特数, None=无损
        phase_align: 是否相位对齐模板到原始信号 (periodic/transient推荐开启)

    Returns:
        重建信号
    """
    if len(retrieval_results) == 0:
        return y_received if y_original is None else y_original

    best_id = retrieval_results[0][0]
    record = kb.get_record(best_id)
    if record is None:
        return y_received if y_original is None else y_original

    y_template = record.raw_data

    # 无原始信号 → 退回直接传
    if y_original is None:
        return y_received

    # 相位对齐 (发射端操作: 以原始信号为基准对齐模板)
    aligned_template = y_template
    if phase_align:
        aligned, lag, corr = phase_align_via_cross_correlation(y_original, y_template)
        if corr > 0.3:
            aligned_template = aligned
        # corr太低则放弃对齐, 用原模板

    # 发射端: residual = y_original - aligned_template
    residual = y_original - aligned_template

    # 残差过信道 (模拟真实传输)
    if channel is not None:
        residual = channel.forward(residual)

    # 量化 (可选)
    if n_bits is not None and n_bits > 0:
        residual, _ = quantize_uniform(residual, n_bits)

    # 接收端: y_recon = aligned_template + residual
    return aligned_template + residual


def compute_residual(
    y_original: np.ndarray,
    y_template: np.ndarray,
) -> np.ndarray:
    """计算原始信号与模板的残差."""
    return y_original - y_template


def bandwidth_bits(
    n_samples: int,
    n_templates: int,
    residual_bits: int = 8,
) -> dict:
    """计算传输所需比特数."""
    template_id_bits = np.ceil(np.log2(n_templates))
    residual_total_bits = n_samples * residual_bits
    total_bits = template_id_bits + residual_total_bits
    bits_per_sample = total_bits / n_samples
    pcm_bits = n_samples * 8
    compression_ratio = pcm_bits / total_bits if total_bits > 0 else float('inf')
    return {
        'template_id_bits': int(template_id_bits),
        'residual_total_bits': residual_total_bits,
        'total_bits': total_bits,
        'bits_per_sample': bits_per_sample,
        'compression_ratio': compression_ratio,
        'pcm_8bit_baseline': pcm_bits,
    }
