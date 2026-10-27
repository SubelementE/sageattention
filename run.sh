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
# Ascend C Kernel 直调运行脚本 - SAGEAttention
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# [MODIFY] 替换为实际的算子二进制名称
OP_NAME="sageattention_custom"

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
fi

# 统一错误输出函数
die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先配置 CANN 环境"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在，请先完整编译"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j4  || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py 执行失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel 运行失败（exit code $?）"
[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在（静默失败）"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "精度验证失败（verify_result.py 返回非零）"

echo "=== 完成 ==="
exit 0
