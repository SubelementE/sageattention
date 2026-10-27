# sageattention 算子

## 基本信息

- **算子名称**：sageattention
- **目标硬件**：Ascend 910B3 (dav-c220)，还有Ascend 310B3 (dav-c220)YOLOv12适配版
- **支持数据类型**：输入/输出 FP16；Q/K 内部 per-token INT8，PV 为 FP16（910B 无 FP8）

## 数学公式

QK 采用与 GPU 一致的 INT8 量化（smoothK + per-token scale，int32 累加后反量化），
PV 保持 FP16（910B 无 FP8 Cube 算力）：

```
输入: Q, K, V ∈ ℝ^{BH×N×d} (FP16)
输出: O ∈ ℝ^{BH×N×d} (FP16)

1. Q 缩放: Q_scaled = Q * 1/√d
2. K 平滑: K_mean = mean(K, dim=1), K_smoothed = K - K_mean
3. 注意力分数: S = Q_scaled @ K_smoothed^T
4. Softmax: P = softmax(S, dim=-1)
5. 输出: O = P @ V
```

## 当前实现要点

| 项目 | 实现 |
|------|------|
| 输入布局 | `[BH, N, dPadded]`，`BH = batch × heads`（目标 shape：BH=256, N=1024, d=64） |
| 多核切分 | head-major 连续区间：`total = BH × nQBlocks`，每核处理一段连续工作项 |
| prologue | 每个 head 每核只做一次 `Ks = K − rowmean(K)` 与 `Vp = V`，直接写入 **L1(TSCM, NZ)** 并常驻，所有 Q 块复用；冗余 ≈1× |
| 主循环 | 固定 `bq=64`、`bkv=256`；bmm1 走 **INT8** Cube（`Q_int8@Ks_int8^T`→int32→逐行×逐列反量化），bmm2 `P@Vp` 走 FP16 Cube |
| Softmax | 在线 softmax：逐块 rowmax → `m_new=max(m_old,rowmax)` → `exp(S−m_new)`，并按 `alpha=exp(m_old−m_new)` 重缩放 O/l |
| 异步流水 | bmm1 深度 3（双客户端 + 3 块 S 缓冲，提前两块提交）；bmm2 延后一拍汇总；P/C 各 2 块 |
| 归一化 | 显式维护 `l = Σ exp(S−m)`（与 GPU 一致），epilogue 逐行除以 l |

## 目录结构

```
sageattention/
├── scripts/        # 测试脚本
│   ├── gen_data.py       # 测试数据生成 (支持 FP16/FP32/BF16)
│   └── verify_result.py  # 精度验证 (自动选择 rtol/atol)
├── sageattention.asc  # 910B Kernel 实现
├── sageattention310.asc  # 310B 端侧 Kernel 实现
├── sageattention310_area.asc  # YOLOv12 Area Attention 310B Kernel

├── CMakeLists.txt  # 构建脚本
├── run.sh          # 运行脚本
├── data_utils.h    # 数据读写工具
└── README.md       # 本文件
```

## 编译运行指南

```bash
# 1. 设置 CANN 环境
source ${ASCEND_HOME_PATH}/set_env.sh

# 2. 编译
cd operators/sageattention
mkdir -p build && cd build
cmake .. && make -j4

# 3. 运行 (包含数据生成 + Kernel 执行 + 精度验证)
cd ..
bash run.sh

# 4. 仅运行 (跳过编译)
bash run.sh --skip-build
```


## 310B 端侧算子（sageattention310）

面向 Atlas 200I/310B 系列的独立算子，源码 [sageattention310.asc](sageattention310.asc)。

| 项 | 910B 版 | 310B 版 |
|------|---------|---------|
| 精度路径 | INT8 QK + FP16 PV | **FP16 QK + FP16 PV** |
| 调度模型 | MIX AIC/AIV + KFC 异步流水 | **单核同步 Matmul**（`Matmul::Init` + `IterateAll`） |
| Ks / V 驻留 | L1(TSCM/NZ) | **GM**（V 直接作为 bmm2 的 B） |
| 缓冲 | S 3 块、P/C 2 块 | **单缓冲**（全同步） |
| Softmax | running max + 显式 l | 同左 |
| 默认 shape | BH=256, N=1024, d=64 | **BH=8, N=512, d=64**（BQ=64, BKV=256） |

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build_310
cmake --build build_310
# 产物: build_310/sageattention310_custom (device 侧 dav-m300)
```

状态：交叉编译通过（CANN 8.2.RC1 的 bisheng 已支持 `dav-m300`）；无 310B 设备，未上板。
310B 不支持 `--cce-aicore-block-local-init`，CMake 已为 310 目标移除该选项。


## YOLOv12 Area Attention（sageattention310_area）

单文件算子，4D 布局的全部逻辑（qkv chunk、area 转置、L 填充/掩码、反变换）都在
[sageattention310_area.asc](sageattention310_area.asc) 内，没有 Python 适配层。

C API：

```c
int sageattention310_area_forward_qkv(
    const void* qkv, void* out, int B, int C3, int H, int W,
    int heads, int area, void* stream);
// qkv: (B, 3*C1, H, W) fp16, C1 = heads*head_dim
// out: (B, C1, H, W) fp16
```

Kernel 内部流程：`sa310_pack_qkv`（4D→[BH,Lpad,64]，Gather 转置）→
`sageattention310_area_custom`（注意力，最后一个 KV 块做 L 掩码）→
`sa310_unpack`（[BH,Lpad,64]→4D）。

| stage | L | Lpad (BKV=128) | Q 块数 (BQ=128) |
|---|---|---|---|
| P3 80×80 | 1600 | 1664 | 13 |
| P4 40×40 | 400 | 512 | 4 |
| P5 20×20 | 100 | 128 | 1 |

编译：

```bash
cmake --build build_310
# build_310/libsageattention310_area.so        （4D C API）
# build_310/sageattention310_area_custom      （standalone 自测）
```

YOLOv12 侧只需一行 ctypes 调用（不再做任何 reshape/pad）：

```python
# qkv = self.qkv(x)  -> (B, 3*C1, H, W) fp16 contiguous
out = torch.empty((B, qkv.shape[1] // 3, H, W), dtype=qkv.dtype, device=qkv.device)
_lib.sageattention310_area_forward_qkv(
    qkv.data_ptr(), out.data_ptr(), B, qkv.shape[1], H, W,
    heads, area, torch.npu.current_stream().npu_stream)
x = self.proj(out)
x = x + self.pe(x)
```

即 `AAttn.forward` 中 `chunk/q.view/permute/SDPA/逆 permute` 全部由算子内部完成。
性能配置：`BQ=128 / BKV=128`，UB ≈ 189KB（310B1 为 248KB）；K/V 重读次数减半。
pack/unpack 的完整 tile 用 16 次 `TransDataTo5HD` 替代 64 次 `Gather`（尾部回退）。\nC API 按 shape 缓存 workspace（最多 8 组），命中时不做 tiling 构建/分配/同步，返回后由调用方
stream 排序；需要释放时调用 `sageattention310_area_release()`。

## API 映射

| 数学操作 | 对应 API | 说明 |
|---------|---------|------|
| Q 缩放 (乘 1/√d) | `Cast` + `Muls<float>` + `Cast` | prologue 中预缩放，物化 `Qpad` |
| K 均值/平滑 | `ReduceSum<float>` + `Adds` | prologue 中按 chunk 逐行平滑，物化 `Ks` |
| bmm1 (QK^T) | `Matmul`（Cube, FP16→FP32） | `S = Qpad @ Ks^T`，M/N/K = 64/256/64 |
| Softmax | `WholeReduceMax` + `Brcb` + `Sub` + `Exp` + `Cast` | 在线 running max + 逐行重缩放 |
| bmm2 (PV) | `Matmul`（Cube, FP16→FP32） | `C = P @ Vp`，M/N/K = 64/80/256 |
| 归一化 | `DataCopy` + `Muls` | 取 `C[:,d]` 为行和，epilogue 逐行相除 |
| 数据搬运 | `DataCopy` / `DataCopyParams` | GM↔UB，`SetFlag/WaitFlag` 同步 |

## 已知限制

| 限制项 | 说明 |
|--------|------|
| 硬件 | 910B3 (dav-c220)；该芯片无 Quantize/AntiQuantize，量化路径未启用 |
| 数据类型 | FP16 输入输出；Q/K 内部 per-token INT8 量化（GPU 对齐），PV 为 FP16；910B 无 FP8，未启用 FP8 PV |
| shape | `d` 为编译期常量（当前 64）；`N` 必须是 64/128/256 的倍数 |
| 功能范围 | 仅 non-causal、Q/K/V 头数相同；不支持 GQA/MQA、变长、dropout |
| 上板状态 | 仅交叉编译通过；`scripts/gen_data.py` 已按 `[BH,N,d]` 生成输入与 golden |

