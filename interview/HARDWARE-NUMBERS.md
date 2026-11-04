# 硬件数量级

分三类：本机实测（能讲「我测到」）、设备表（数据手册算术，只能讲「按手册」）、不要背的数。估算时先说数字属于哪一类。
数据中心 GPU、NVLink、HBM 的数量级在 mlsys-infra [HARDWARE-NUMBERS](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/interview/HARDWARE-NUMBERS.md)。核对日期 2026-11-03。

## 十个要背下来的数

| # | 数 | 来源 |
| ---: | --- | --- |
| 1 | GEMV 强度 ≈ 2 / 每权重字节：fp16 1，int8 2，Q4_0 3.6 FLOP/B | E3 |
| 2 | 本机 fp32 拐点 11.5 FLOP/B，int8 29.5 op/B | 实测，E1 |
| 3 | 单核 DRAM 只读 13.4 GB/s，4 核 48.4（3.6 倍） | 实测，E2 |
| 4 | int8 VNNI 对 fp32 FMA：按位宽 4 倍，实测 2.56 倍 | 实测 |
| 5 | Q8_0 8.5、Q4_0 4.5、Q4_1 5.0 bit/权重 | E25 |
| 6 | Llama-7B Q4_0 3.79 GB；Pi 5 上限 4.5 tok/s，Orin Nano 17.9 | E25 + 设备表 |
| 7 | MCU SRAM 256 KB–1 MB；MobileNetV2 1.0@224 激活下界 1.44 MiB，放不下 | 设备表 + E10 |
| 8 | fp32 depthwise 3×3 强度 4.5 FLOP/B | E21 |
| 9 | 8 → 4 bit，SQNR 少约 24 dB | E27 |
| 10 | maddubs 不饱和要求 \|w\| ≤ 64 | E6 |

## 本机实测（Ice Lake 服务器虚拟机，4 物理核 / 8 逻辑核，AVX-512 + VNNI）

来自 [host-roofline](../solutions/edge-runtime/host-roofline/README.md) 的 `artifacts/raw/summary.json`，由 `claims.json` 的 `t1.*`、`t4.*` 校验。

| 量 | 1 线程 | 4 线程 | 备注 |
| --- | ---: | ---: | --- |
| fp32 FMA 峰值 | 154.9 GFLOP/s | 596.6 | zmm FMA，折合约 2.42 GHz × 2 端口 × 16 lane × 2 |
| int8 VNNI 峰值 | 396.1 GOP/s | 1492.7 | 是 fp32 的 2.56 / 2.50 倍 |
| L1 读 | 166 GB/s | 596 | 4 线程要线程内重复扫描、自旋线程池才测得到 |
| DRAM 只读 | 13.4 GB/s | 48.4 | 单核受在途请求数限制 |
| DRAM triad | 13.8（按写分配计 18.4） | 49.1 | |
| fp32 拐点 | 11.5 FLOP/B | 12.3 | |
| int8 拐点 | 29.5 op/B | 30.8 | |

其他实验里会被追问的实测数：

| 量 | 数 | 实验 |
| --- | --- | --- |
| 打包 VNNI int8 GEMM | 比逐点积 VNNI 快 2.0–30.8 倍 | [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) |
| ORT int8 对 fp32 | 1 线程 1.39–3.08 倍；4 线程 0.76–2.14 倍 | int8-kernels |
| depthwise NHWC + SIMD | 10.4–11.4 GFLOP/s，CHW 的 8–9 倍 | [conv-kernels](../solutions/edge-runtime/conv-kernels/README.md) |
| 单线程最好的 256³ fp32 GEMM 调度 | 85.7 GFLOP/s（numpy 90.1） | [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md) |
| 7B 形状 Q4 × Q8 GEMV | 1 线程 1034 µs（9.1 GB/s），4 线程 311 µs | [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) |
| 跨线程握手 | 约 0.8 µs | [dma-double-buffer](../solutions/edge-runtime/dma-double-buffer/README.md) |

机器边界：虚拟机里没有性能计数器，端口压力、缓存命中都是推算；时钟按 fp32 峰值反推约 2.42 GHz（系统报告 2.8 GHz）。后台服务偶尔占一个核，计时前要确认空闲。

## 服务器实测（Xeon 8470Q + RTX 6000D）

同一份代码在 AutoDL 服务器上重跑，全表见 [CROSS-HOST](CROSS-HOST.md)，分析见 [`reports/edge-runtime-server`](../reports/edge-runtime-server/edge_runtime_server_report.pdf)。
`claims.json` 中 `srv-*`、`gpu-*` 条目校验。

| 量 | 数 | 和本地比 |
| --- | --- | --- |
| 1 线程 DRAM 只读 | 17.5 GB/s | 只高 1.30 倍：单核带宽受延迟限制，不随「服务器更强」上涨 |
| 16 线程 DRAM 只读 | 193.5 GB/s | 单插槽 8 × DDR5-4800 理论 307 的 63%，8 → 16 线程只涨 1.6 倍 |
| fp32 FMA 1 / 16 线程 | 232 / 3167 GFLOP/s | 反推单核 3.6 GHz、16 核 3.1 GHz |
| 拐点 1 → 16 线程 | 13.3 → 16.4 FLOP/B | 算力扩展快于带宽，多核让更多层变成带宽受限 |
| AMX int8 512³ | 1704 GOP/s | VNNI 打包 563 的 3.0 倍；单核峰值约 4.3 TOP/s 的 40% |
| AMX，K = 24 | 162 GOP/s | 补零到 64，是 VNNI（270）的 0.6 倍 |
| RTX 6000D 读 / 拷贝 | 1333 / 1280 GB/s | 4 GB 顺序；规格 1398 |
| RTX 6000D L2 | 112 MB | 33 MB 的权重会常驻，计时必须轮换拷贝 |
| Triton eager 下发 | 约 22 µs / 次 | CUDA graph 后 3.7 µs |
| Qwen2.5-7B 只算 Linear 解码 | fp16 92、Q8 163、Q4 284 tok/s（graph） | Q4 eager 只有 176 |
| llama.cpp Q4_0 7B 1 线程解码 | 3.97 tok/s | × 每 token 4.1 GB 权重 = 16.3 GB/s，单核 DRAM 读的 93% |

## 设备表（数据手册算术，不是实测）

来自 `edge_harness/devices.py`。带宽是数据率 × 位宽，不是持续值。

| 设备 | CPU | 主频 | SRAM / Flash | DRAM 峰值 |
| --- | --- | ---: | --- | ---: |
| nRF52840（Arduino Nano 33 BLE Sense，HarvardX TinyML 套件） | Cortex-M4F | 64 MHz | 256 KB / 1 MB | 无外部 DRAM |
| STM32H747（Portenta H7，CS249r 2023） | Cortex-M7 + M4 | 480 MHz | 1 MB（分多个域） / 2 MB | 无 |
| Raspberry Pi 5 | 4 × Cortex-A76 | 2.4 GHz | | 17.1 GB/s（LPDDR4X-4267，32 位） |
| Jetson Orin Nano 8GB | 6 × Cortex-A78AE + Ampere GPU | 1.5 GHz | | 68 GB/s（128 位 LPDDR5；4GB 版 64 位） |

代入公式的例子：

| 场景 | 数 | 算法 |
| --- | --- | --- |
| 64 MHz MCU、32 位总线 | 256 MB/s | `64e6 × 4`，零等待、每周期一次访问的上限 |
| LPDDR4X-4266、64 位 | 34.1 GB/s | `4266e6 × 8` |
| Q4_0、5×10^8 参数、上面这条 LPDDR | 8.24 ms/token | 2.81×10^8 B / 34.1 GB/s，只算权重 |
| Pi 5 跑 GPT-2 124M fp32 | ≥ 29 ms/token | 0.50 GB / 17.1 GB/s，单核实际更慢（E2） |

## 不要直接背的数

- 厂商 INT8 TOPS：常把 2:4 稀疏算进去，稠密是一半，还没算利用率。本仓库的设备表故意不存 TOPS，需要时作为参数传入。
- 总线峰值当持续带宽：本机单核只读只有 4 核的 28%，Pi 5 的 17.1 也不是单个 A76 能用到的数。
- 「int8 是 fp32 的 4 倍」：本机峰值 2.56 倍；depthwise 上 ORT 4 线程 int8 比 fp32 还慢。

## 和 mlsys-infra 的对照

| | 数据中心 GPU（mlsys-infra） | 端侧（本仓库） |
| --- | --- | --- |
| 权重带宽 | HBM，TB/s | LPDDR 17–68 GB/s；MCU 是 SRAM，几百 MB/s |
| 拐点 | H100 bf16 约 295 FLOP/B | 本机 fp32 11.5、int8 29.5 |
| 先撞上的容量 | HBM 里的权重和 KV | MCU 上是 SRAM 里的激活；手机上经常仍是权重 |
| decode 受限于 | batch 小时是 HBM 带宽，batch 大时是算力（F16） | 几乎总是 DRAM 带宽，Q4 单核时是每块指令数 |
