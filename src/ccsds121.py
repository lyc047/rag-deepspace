"""CCSDS 121.0-B-3 无损数据压缩编码器 (简化实现).

标准流程:
1. 预处理器: 单位延迟预测器 x[n] -> delta[n] = x[n] - x[n-1]
   每J个样本插入一个参考样本(直接传原始值)
2. 块自适应熵编码: J样本/块, 每块选最优Rice参数k
   编码: quotient = floor(|delta|/2^k) [unary], remainder = |delta| mod 2^k [k-bit], sign [1-bit]

用于压缩比对比实验, 非完整CCSDS兼容实现.
"""
import numpy as np
from typing import Tuple


def ccsds121_preprocess(data: np.ndarray, j: int = 16) -> np.ndarray:
    """单位延迟预测器.

    delta[n] = data[n] - data[n-1]  (n不是J的倍数)
    delta[n] = data[n]              (n是J的倍数, 参考样本)

    Args:
        data: 输入数据 (float, 将转为整数)
        j: 参考样本间隔

    Returns:
        预处理后的整数残差数组
    """
    # 转为整数 (假设输入已量化)
    x = np.round(data).astype(np.int64)
    n = len(x)
    delta = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if i % j == 0:
            delta[i] = x[i]  # 参考样本
        else:
            delta[i] = x[i] - x[i - 1]

    return delta


def rice_encode_block(block: np.ndarray) -> Tuple[int, int]:
    """对J样本块用Rice编码, 返回(总比特数, 最优k).

    Args:
        block: 预处理后的残差数组 (参考样本除外, 应已在外部处理)

    Returns:
        (total_bits, best_k)
    """
    j = len(block)
    best_bits = float('inf')
    best_k = 0

    # 搜索最优k (0到14, 对应CCSDS标准)
    for k in range(15):
        bits = 0
        for val in block:
            if val == 0:
                # quotient=0 -> 1 bit (just the stop bit)
                bits += 1 + k + 1  # unary stop + remainder + sign
            else:
                abs_val = abs(val)
                quotient = abs_val >> k  # floor(|val| / 2^k)
                # Unary: quotient个0 + 1个1作为停止位
                bits += quotient + 1  # unary
                bits += k             # remainder (k bits)
                bits += 1             # sign bit

        if bits < best_bits:
            best_bits = bits
            best_k = k

    return best_bits, best_k


def ccsds121_compress(data: np.ndarray, j: int = 16) -> dict:
    """CCSDS 121.0-B-3 压缩 (仅计算比特数, 不实际编码).

    Args:
        data: 输入信号
        j: 块大小

    Returns:
        dict: {
            'original_bits': 原始比特数 (假设8-bit量化),
            'compressed_bits': 压缩后比特数,
            'compression_ratio': original/compressed,
            'bits_per_sample': compressed/N,
            'reference_bits': 参考样本消耗的比特,
            'block_k_values': 每块选的最优k值列表,
        }
    """
    # 先量化到8-bit整数
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < 1e-10:
        # 恒定信号, 几乎无信息
        return {
            'original_bits': len(data) * 8,
            'compressed_bits': len(data) * 1,
            'compression_ratio': 8.0,
            'bits_per_sample': 1.0,
            'reference_bits': 0,
            'block_k_values': [],
        }

    # 量化: 线性映射到 [0, 255]
    data_norm = np.round((data - data_min) / (data_max - data_min) * 255).astype(np.int64)

    # 预处理器
    n = len(data_norm)
    delta = ccsds121_preprocess(data_norm, j)

    total_bits = 0
    ref_bits = 0
    k_values = []

    # 逐块编码
    for start in range(0, n, j):
        end = min(start + j, n)
        block = delta[start:end]

        # 第一个样本是参考样本 (8-bit直接传)
        ref_bits += 8
        total_bits += 8

        if len(block) > 1:
            b, k = rice_encode_block(block[1:])  # 跳过分隔样本
            total_bits += b
            k_values.append(k)

    # 加上数据范围信息 (min, max 各32-bit)
    total_bits += 64

    original_bits = n * 8
    compression_ratio = original_bits / total_bits if total_bits > 0 else float('inf')

    return {
        'original_bits': original_bits,
        'compressed_bits': total_bits,
        'compression_ratio': compression_ratio,
        'bits_per_sample': total_bits / n,
        'reference_bits': ref_bits,
        'block_k_values': k_values,
    }


def ccsds121_decompress(compressed_bits: int, data_min: float, data_max: float,
                        delta: np.ndarray, j: int = 16) -> np.ndarray:
    """CCSDS 121.0解压 (简化版——直接使用已知的delta)."""
    n = len(delta)
    x = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if i % j == 0:
            x[i] = delta[i]
        else:
            x[i] = x[i - 1] + delta[i]

    # 反量化
    data_float = x.astype(np.float64) / 255.0 * (data_max - data_min) + data_min
    return data_float


def benchmark_ccsds(data: np.ndarray) -> dict:
    """快速基准测试: 对数据跑CCSDS 121.0压缩并返回统计."""
    return ccsds121_compress(data)
