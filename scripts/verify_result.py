# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 结果验证脚本 - SAGEAttention (支持 FP16/FP32/BF16)
# ============================================================================

import numpy as np
import sys

# P0-3/P1-5: 根据 dtype 自动选择 rtol/atol 和加载方式
DTYPE_CONFIGS = {
    "fp32": {"dtype": np.float32, "rtol": 1e-5, "atol": 1e-5},
    "fp16": {"dtype": np.float16, "rtol": 1e-2, "atol": 1e-2},
    "bf16": {"dtype": np.uint16, "rtol": 1e-2, "atol": 1e-2},  # P1-5: BF16 使用 uint16 加载
}

def load_bf16_as_float32(path):
    """P1-5 修复: BF16 数据以 uint16 存储, 需要正确转换为 float32 进行比对"""
    raw = np.fromfile(path, dtype=np.uint16)
    import struct
    # BF16: 高 16 位是 BF16, 低 16 位补 0 转为 FP32
    fp32_values = np.zeros(len(raw), dtype=np.float32)
    for i in range(len(raw)):
        # BF16 占 16 位, 转为 FP32 需要在低位补 16 个 0
        fp32_bytes = struct.pack('<I', int(raw[i]) << 16)
        fp32_values[i] = struct.unpack('<f', fp32_bytes)[0]
    return fp32_values

def verify_result(output_path, golden_path, dtype_str="fp16"):
    config = DTYPE_CONFIGS.get(dtype_str, DTYPE_CONFIGS["fp16"])
    dtype = config["dtype"]
    rtol = config["rtol"]
    atol = config["atol"]

    # P1-5 修复: BF16 使用专用加载函数
    if dtype_str == "bf16":
        output = load_bf16_as_float32(output_path)
        golden = load_bf16_as_float32(golden_path)
    else:
        output = np.fromfile(output_path, dtype=dtype)
        golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    # 转为 float32 进行精度比对
    output_f32 = output.astype(np.float32)
    golden_f32 = golden.astype(np.float32)

    if np.allclose(output_f32, golden_f32, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}, dtype={dtype_str}")
        print(f"Max diff: {np.max(np.abs(output_f32 - golden_f32))}")
        print(f"Mean diff: {np.mean(np.abs(output_f32 - golden_f32))}")
        return True
    else:
        diff = np.abs(output_f32 - golden_f32)
        print(f"Verification FAILED! dtype={dtype_str}")
        print(f"Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden_f32))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        # 打印前 10 个差异
        if len(mismatches) > 0:
            for idx in mismatches[:10]:
                print(f"  [{idx}] output={output[idx]}, golden={golden[idx]}, diff={diff[idx]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin> [dtype]")
        print("  dtype: fp32, fp16 (default), bf16")
        sys.exit(1)

    dtype_str = sys.argv[3] if len(sys.argv) > 3 else "fp16"
    success = verify_result(sys.argv[1], sys.argv[2], dtype_str)
    sys.exit(0 if success else 1)
