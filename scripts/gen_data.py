# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 测试数据生成脚本 - SAGEAttention (多 head)
#
# 数据布局与 kernel 一致: q/k/v/golden 均为 [BH, N, dPadded] 连续存储,
# 其中 BH = batch * heads。默认对齐当前 kernel 的编译期参数:
#   BH=256 (batch=8, heads=32), N=1024, d=64
#
# 用法:
#   python gen_data.py [fp16|fp32|bf16] [BH] [N] [d]
# ============================================================================

import os
import sys

import numpy as np

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

dtype_str = sys.argv[1] if len(sys.argv) > 1 else "fp16"
BH = int(sys.argv[2]) if len(sys.argv) > 2 else 256
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
d = int(sys.argv[4]) if len(sys.argv) > 4 else 64

if dtype_str not in ("fp16", "fp32", "bf16"):
    print(f"Unsupported dtype: {dtype_str}. Use fp32, fp16, or bf16.")
    sys.exit(1)

dPadded = ((d + 15) // 16) * 16
total_elements = BH * N * dPadded

# ---- 生成输入: [BH, N, d]，并补零到 dPadded --------------------------------
np.random.seed(42)
q = np.random.randn(BH, N, d).astype(np.float32)
k = np.random.randn(BH, N, d).astype(np.float32)
v = np.random.randn(BH, N, d).astype(np.float32)

q_padded = np.zeros((BH, N, dPadded), dtype=np.float32)
k_padded = np.zeros((BH, N, dPadded), dtype=np.float32)
v_padded = np.zeros((BH, N, dPadded), dtype=np.float32)
q_padded[:, :, :d] = q
k_padded[:, :, :d] = k
v_padded[:, :, :d] = v
del q, k, v

if dtype_str == "bf16":
    # BF16: 用 float32 截断模拟
    def to_bf16(a):
        return (a.view(np.uint32) >> 16).astype(np.uint16)
    q_final = to_bf16(q_padded)
    k_final = to_bf16(k_padded)
    v_final = to_bf16(v_padded)
elif dtype_str == "fp32":
    q_final, k_final, v_final = q_padded, k_padded, v_padded
else:
    q_final = q_padded.astype(np.float16)
    k_final = k_padded.astype(np.float16)
    v_final = v_padded.astype(np.float16)

q_final.tofile("input/input_q.bin")
k_final.tofile("input/input_k.bin")
v_final.tofile("input/input_v.bin")

# ---- golden: O = softmax((Q/√d) @ K_s^T) @ V -------------------------------
# kernel 的高效路径: K 先逐行减均值 (smoothK)，Q 预乘 1/√d。
# 逐 head 计算 (每个 head 的 N×N 中间矩阵约 4 MB)，避免 BH×N×N 撑爆内存。
inv_sqrt_d = 1.0 / np.sqrt(d)
q_scaled = q_padded * inv_sqrt_d
k_smooth = k_padded.copy()
k_smooth[:, :, :d] -= k_smooth[:, :, :d].mean(axis=2, keepdims=True)
v_valid = v_padded[:, :, :d]

golden = np.zeros((BH, N, dPadded), dtype=np.float32)
for bh in range(BH):
    scores = q_scaled[bh, :, :d] @ k_smooth[bh, :, :d].T          # [N, N]
    scores -= scores.max(axis=1, keepdims=True)                   # 数值稳定
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)                     # softmax
    golden[bh, :, :d] = probs @ v_valid[bh]
    if (bh + 1) % 64 == 0:
        print(f"  golden: {bh + 1}/{BH} heads")

if dtype_str == "bf16":
    golden_out = (golden.view(np.uint32) >> 16).astype(np.uint16)
elif dtype_str == "fp32":
    golden_out = golden.astype(np.float32)
else:
    golden_out = golden.astype(np.float16)

golden_out.tofile("output/golden.bin")

print(f"Generated test data: BH={BH}, N={N}, d={d}, dPadded={dPadded}, "
      f"elements={total_elements}, dtype={dtype_str}")
print(f"  input/input_q.bin: {q_final.shape}, {q_final.dtype}")
print(f"  input/input_k.bin: {k_final.shape}, {k_final.dtype}")
print(f"  input/input_v.bin: {v_final.shape}, {v_final.dtype}")
print(f"  output/golden.bin: {golden_out.shape}, {golden_out.dtype}")
print(f"  inv_sqrt_d = {inv_sqrt_d}")
print("  Kernel: O = softmax((Q/sqrt(d)) @ (K - rowmean(K))^T) @ V")
