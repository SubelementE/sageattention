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
# CMake 构建配置 - SAGEAttention Kernel 直调 (Cube/Matmul 重写版)
# ============================================================================
#
# 本工程把 host 与 device 放在同一个 .asc 中, 由 AscendC 编译器 (ccec/bisheng)
# 的单源机制自动划分 host / device 两遍. 该用法与 CANN 自带的直调样例
# (ascendc/act/examples) 一致, 因此这里直接以 custom command 驱动编译器, 而不
# 依赖 x86_64-linux/tikcpp/ascendc_kernel_cmake 的 ASC 语言模块 (本机 CANN
# 8.2.RC1 自带的该模块既不会注入 -xcce, 其 -arch Ascend910Bx 也会被 bisheng
# 忽略, 导致代码被当作 dav-m100 编译而报错).
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# CANN toolkit root: prefer the sourced env, fall back to the `latest` symlink.
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_DIR "$ENV{ASCEND_HOME_PATH}")
else()
    set(ASCEND_HOME_DIR "/usr/local/Ascend/ascend-toolkit/latest")
endif()

project(sageattention_custom LANGUAGES CXX)

# ---- AscendC compiler (ccec is a symlink of bisheng) -----------------------
set(SA_CCEC_BIN_DIR "${ASCEND_HOME_DIR}/compiler/ccec_compiler/bin")
find_program(SA_CCEC NAMES ccec bisheng PATHS "${SA_CCEC_BIN_DIR}" NO_DEFAULT_PATH)
if(NOT SA_CCEC)
    find_program(SA_CCEC NAMES ccec bisheng)
endif()
if(NOT SA_CCEC)
    message(FATAL_ERROR "AscendC compiler (ccec/bisheng) not found under ${SA_CCEC_BIN_DIR}")
endif()

# ---- NPU arch: dav-c220 == Ascend910B/910B3 (Atlas A2) ---------------------
# 其他取值示例: dav-3510 (Atlas A5)。覆盖: -DSA_NPU_ARCH=dav-xxxx
set(SA_NPU_ARCH "dav-c220" CACHE STRING "AscendC aicore arch (dav-c220 for 910B)")

set(SA_TIKCPP  "${ASCEND_HOME_DIR}/compiler/tikcpp")
set(SA_ASC_INC "${ASCEND_HOME_DIR}/x86_64-linux/ascendc/include")
set(SA_LIB_DIR "${ASCEND_HOME_DIR}/x86_64-linux/lib64")

# ---- include dirs for a mixed host+device direct-call kernel ---------------
set(SA_INCLUDES
    ${SA_TIKCPP}
    ${SA_TIKCPP}/tikcfw
    ${SA_TIKCPP}/tikcfw/impl
    ${SA_TIKCPP}/tikcfw/interface
    ${ASCEND_HOME_DIR}/include
    ${ASCEND_HOME_DIR}/include/experiment
    ${SA_ASC_INC}/highlevel_api
    ${SA_ASC_INC}/basic_api
    ${SA_ASC_INC}/highlevel_api/impl
    ${SA_ASC_INC}/basic_api/impl
    ${SA_ASC_INC}/basic_api/interface
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ---- host libs the tiling/platform runtime resolves against ---------------
# 注意: 8.2.RC1 没有 libunified_dlog; 其 CheckLogLevel 由 libascendalog 提供.
set(SA_LIBS
    ascendcl runtime tiling_api register platform ascendalog graph_base m dl c_sec stdc++
)

# ---- device (aicore) flags -------------------------------------------------
set(SA_COMPILE_FLAGS
    -O2
    -std=c++17
    -xcce
    --cce-aicore-arch=${SA_NPU_ARCH}
    --cce-aicore-block-local-init
    --cce-auto-sync
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -Wno-macro-redefined
    -Wno-ignored-attributes
    # Cube/Matmul 直调算子: kernel 有 workspace 形参, tiling 为最后一个形参
    -DHAVE_TILING
    -DHAVE_WORKSPACE
)

set(SA_INC_FLAGS "")
foreach(d IN LISTS SA_INCLUDES)
    list(APPEND SA_INC_FLAGS -I${d})
endforeach()

set(SA_LIB_FLAGS "")
foreach(l IN LISTS SA_LIBS)
    list(APPEND SA_LIB_FLAGS -l${l})
endforeach()

set(SA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sageattention.asc")
set(SA_OUT "${CMAKE_BINARY_DIR}/sageattention_custom")

add_custom_command(
    OUTPUT "${SA_OUT}"
    COMMAND ${SA_CCEC} ${SA_COMPILE_FLAGS} ${SA_INC_FLAGS} -L${SA_LIB_DIR}
            "${SA_SRC}" -o "${SA_OUT}" ${SA_LIB_FLAGS}
    DEPENDS "${SA_SRC}"
    COMMENT "Compiling AscendC direct-call kernel ${SA_SRC} -> ${SA_OUT}"
    VERBATIM
)

# Note: the target name differs from the output file on purpose -- naming the
# target after the produced file makes Makefile generators warn about a
# circular dependency.
add_custom_target(sageattention_kernel ALL DEPENDS "${SA_OUT}")

message(STATUS "AscendC compiler : ${SA_CCEC}")
message(STATUS "AscendC arch     : ${SA_NPU_ARCH}")
message(STATUS "CANN toolkit     : ${ASCEND_HOME_DIR}")

# ============================================================================
# Ascend 310B (dav-m300) edge variant -- separate kernel, separate target.
# 310B has no KFC/MIX AIC-AIV; the kernel uses synchronous Matmul + FP16.
# ============================================================================
set(SA310_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sageattention310.asc")
set(SA310_OUT "${CMAKE_BINARY_DIR}/sageattention310_custom")

string(REPLACE "--cce-aicore-arch=${SA_NPU_ARCH}" "--cce-aicore-arch=dav-m300"
       SA310_COMPILE_FLAGS "${SA_COMPILE_FLAGS}")
# dav-m300 does not support block-local variable initialization.
string(REPLACE "--cce-aicore-block-local-init" "" SA310_COMPILE_FLAGS "${SA310_COMPILE_FLAGS}")

add_custom_command(
    OUTPUT "${SA310_OUT}"
    COMMAND ${SA_CCEC} ${SA310_COMPILE_FLAGS} ${SA_INC_FLAGS} -L${SA_LIB_DIR}
            "${SA310_SRC}" -o "${SA310_OUT}" ${SA_LIB_FLAGS}
    DEPENDS "${SA310_SRC}"
    COMMENT "Compiling AscendC 310B kernel ${SA310_SRC} -> ${SA310_OUT}"
    VERBATIM
)
add_custom_target(sageattention310_kernel ALL DEPENDS "${SA310_OUT}")

message(STATUS "AscendC 310B arch: dav-m300")

# ---- YOLOv12 area-attention variant (also dav-m300) -----------------------
set(SA310A_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sageattention310_area.asc")
set(SA310A_OUT "${CMAKE_BINARY_DIR}/sageattention310_area_custom")
add_custom_command(
    OUTPUT "${SA310A_OUT}"
    COMMAND ${SA_CCEC} ${SA310_COMPILE_FLAGS} ${SA_INC_FLAGS} -L${SA_LIB_DIR}
            "${SA310A_SRC}" -o "${SA310A_OUT}" ${SA_LIB_FLAGS}
    DEPENDS "${SA310A_SRC}"
    COMMENT "Compiling AscendC 310B area-attention kernel ${SA310A_SRC} -> ${SA310A_OUT}"
    VERBATIM
)
add_custom_target(sageattention310_area_kernel ALL DEPENDS "${SA310A_OUT}")

# ---- shared library exposing the single-call 4D API -----------------------
set(SA310A_LIB "${CMAKE_BINARY_DIR}/libsageattention310_area.so")
add_custom_command(
    OUTPUT "${SA310A_LIB}"
    COMMAND ${SA_CCEC} ${SA310_COMPILE_FLAGS} -fPIC -shared ${SA_INC_FLAGS} -L${SA_LIB_DIR}
            "${SA310A_SRC}" -o "${SA310A_LIB}" ${SA_LIB_FLAGS}
    DEPENDS "${SA310A_SRC}"
    COMMENT "Building 310B area-attention shared library ${SA310A_LIB}"
    VERBATIM
)
add_custom_target(sageattention310_area_lib ALL DEPENDS "${SA310A_LIB}")
