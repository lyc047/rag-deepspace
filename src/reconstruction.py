"""信号重建模块：模板匹配 + 残差量化传输.

Phase 3: 支持残差量化，模拟实际通信中的比特约束。
"""
import numpy as np
from typing import List, Tuple, Optional
from knowledge import KnowledgeBase


def quantize_uniform(
    x: np.ndarray,
    n_bits: int,
    max_val: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """均匀量化器.

    Args:
        x: 输入信号
        n_bits: 量化比特数 (1-16)
        max_val: 量化范围 [-max_val, max_val]。None时自动使用|x|的最大值。

    Returns:
        (x_quantized, step_size)
    """
    n_bits = max(1, min(n_bits, 16))
    if max_val is None:
        max_val = float(np.max(np.abs(x)))
    if max_val < 1e-12:
        return np.zeros_like(x), 0.0

    levels = 2 ** n_bits - 1
    step = 2.0 * max_val / levels
    # 量化 + 反量化
    x_clipped = np.clip(x, -max_val, max_val)
    x_quantized = np.round(x_clipped / step) * step
    return x_quantized, step


def reconstruct_from_template(
    y_query: np.ndarray,
    retrieval_results: List[Tuple[int, float]],
    kb: KnowledgeBase,
    n_bits: Optional[int] = None,
    y_original: Optional[np.ndarray] = None,
) -> np.ndarray:
    """基于检索到的最佳模板重建信号.

    Args:
        y_query: 接收到的含噪信号
        retrieval_results: [(template_id, distance), ...]
        kb: 知识库
        n_bits: 残差量化比特数。None=无损(等于y_received), 0=纯模板
        y_original: 原始干净信号（用于计算发射端残差）。None时用y_query

    Returns:
        重建信号
    """
    if len(retrieval_results) == 0:
        return np.zeros_like(y_query) if n_bits is None or n_bits > 0 else y_query

    best_id = retrieval_results[0][0]
    record = kb.get_record(best_id)
    if record is None:
        return np.zeros_like(y_query) if n_bits is not None and n_bits == 0 else y_query

    y_template = record.raw_data

    if n_bits is None:
        # 无损模式：y_recon = y_template + (y_query - y_template) = y_query
        return y_query

    if n_bits == 0:
        # 纯模板模式（Phase 1 MVP）：不传残差
        return y_template.copy()

    # 量化模式：模拟发射端计算残差并量化
    y_clean = y_original if y_original is not None else y_query
    residual = y_clean - y_template
    residual_q, _ = quantize_uniform(residual, n_bits)
    return y_template + residual_q


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
    """计算传输所需比特数.

    Args:
        n_samples: 信号样本数
        n_templates: 知识库模板总数
        residual_bits: 残差量化比特数

    Returns:
        dict with keys: template_id_bits, residual_total_bits, total_bits,
                        bits_per_sample, compression_ratio
    """
    template_id_bits = np.ceil(np.log2(n_templates))
    residual_total_bits = n_samples * residual_bits
    total_bits = template_id_bits + residual_total_bits
    bits_per_sample = total_bits / n_samples

    # 直接PCM传输（8-bit）作为基准
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
