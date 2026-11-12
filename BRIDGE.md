# 与 mlsys-infra 的对照

[mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra) 回答数据中心上的「算得过来、放得下、传得过来」，本仓库回答同样三个问题在 CPU、MCU、NPU 上的版本。两个仓库共用一套实验约定，公式按同一套推导写，面试时可以从任一侧切到另一侧。

## 共用的约定

| 约定 | mlsys-infra | edge-deploy |
| --- | --- | --- |
| 结果文件 | `mlsys_harness.writer`：JSONL，第一行 `kind: run` 记 commit、平台、argv | `edge_harness.writer`：同一格式，读者用 `read_rows` 跳过首行 |
| 计时 | `mlsys_harness.timing`：p10/p50/p90 | `edge_harness.timing` 同名同义，另加 p99；C++ 侧 `edgebench.h` 的 `time_fn` 输出同样的字段 |
| 发布数字的校验 | `claims.json` + `scripts/check_claims.py` | 同一 spec 格式（`file`、`where`、`field`、`expect`/`min`/`max`、`type: ratio`），`edge_harness.claims` 实现 |
| 仓库检查 | `scripts/check-repo.ps1`：笔记、链接、lab 布局、claims、密钥 | 同一脚本结构，lab 布局要求相同（README、tests、scripts/run_experiments.py、artifacts/raw、artifacts/figures） |
| 报告 | `reports/<lab>/*.tex` + PDF，共用 preamble；`scripts/check_reports.py` 查 PDF、图和参考文献 | 同一套 preamble 风格（追问框、结论框、不能推出框）和同一个检查脚本 |
| 笔记结构 | `notes/check_notes.py`：16 个必备章节、追问链 ≥ 5、≥ 180 行 | 同一校验器，另要求每篇在「关联笔记」里链到 mlsys-infra 的对应模块 |
| GPT-2 困惑度 | [`compiler-compression/gpu/llm_utils.py`](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/gpu/llm_utils.py)：wikitext-2 test、`"\n\n"` 拼接、1024 token 窗口、固定种子 randperm | `q4-gemv/scripts/run_experiments.py` 照搬同一协议，fp32 基线可以直接对表 |

## 模块对照

| edge-deploy | 同一个问题在数据中心 | 换了什么 |
| --- | --- | --- |
| [M01 端侧 roofline](notes/module-01-edge-roofline.md) | [M01 性能模型](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-01-performance-models.md)、[M02 多核与 SIMD](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-02-multicore-simd-threads.md) | 拐点从 H100 的 295 FLOP/byte 换成本机实测的 L1/L2/DRAM 三条屋顶；int8 VNNI 是单独一条算力屋顶 |
| [M02 整数契约](notes/module-02-integer-contract.md) | [M17 压缩与量化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)、[M07 硬件感知 GEMM](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-07-matmul-hardware-aware.md) | 那边是 fake-quant 和 Tensor Core 的精度；这里是 int32 累加、定点 requant 的逐位语义，以及 `vpmaddubsw` 的 int16 饱和 |
| [M03 校准与误差定位](notes/module-03-calibration-localization.md) | [M17 压缩与量化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md) | 那边处理 LLM 激活的离群通道（LLM.int8、SmoothQuant）；这里是 CNN 的 ReLU6 零值尖峰让 KL 校准失效，以及按敏感度的混合精度 |
| [M04 静态内存规划](notes/module-04-static-memory-planning.md) | [M14 显存优化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-14-memory-optimization.md) | 训练侧在时间维重计算激活；这里在空间维按 patch 重算 halo。两边都先找峰值在哪 |
| [M05 卷积实现](notes/module-05-convolution-kernels.md) | [M07 硬件感知 GEMM](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-07-matmul-hardware-aware.md) | GPU 上卷积几乎都走 implicit GEMM；CPU 上 im2col、implicit GEMM、depthwise 专用核、Winograd 各有适用区间 |
| [M06 调度搜索](notes/module-06-schedule-search.md) | [M10 ML 编译器](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-10-ml-compilers.md) | 那边讲融合和 torch.compile；这里在真机上跑 TVM 式的搜索，比较网格、随机、进化和代价模型 |
| [M07 DMA 与双缓冲](notes/module-07-dma-double-buffering.md) | [M09 加速器](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-09-accelerators.md)、[M13 流水线并行](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-13-tensor-pipeline-parallel.md) | 流水线气泡 (p−1)/(m+p−1) 和双缓冲的 min(T_dma, T_comp) 是同一个填充-排空模型 |
| [M08 块量化与解码](notes/module-08-block-quantized-decode.md) | [M15 推理与 KV](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-15-inference-kv-speculative.md)、[M17 压缩与量化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md) | 那边用 GPTQ/AWQ 求权重、在 GPU 上测 decode；这里实现 llama.cpp 的块格式和 CPU 融合 GEMV，困惑度用同一协议；再用 llama.cpp 在同一张 RTX 6000D 上跑同一个 Qwen2.5-7B |

## 实验对照

| edge-deploy 实验 | mlsys-infra 实验 | 能直接对表的数 |
| --- | --- | --- |
| [int8-kernels](solutions/edge-runtime/int8-kernels/README.md) | [CS149 A4 加速器](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cs149/asst4-dnn-accelerator/README.md) | 同一份 GEMM 分块推导；那边是 GPU SGEMM 阶梯，这里是 AVX-512 fp32 与 VNNI int8 阶梯 |
| [lab2-quantization](solutions/mit6.5940/lab2-quantization/README.md) | [15-849 编译与压缩](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/README.md) | 两边都报 per-tensor vs per-channel 的误差；那边的离群通道实验与这里的 KL 零值尖峰是两种「范围被少数值主导」 |
| [q4-gemv](solutions/edge-runtime/q4-gemv/README.md) | [15-849 编译与压缩](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/README.md)、[CS229S P2](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cs229s/project2-kv-speculative/README.md) | GPT-2 fp32 困惑度同一协议；那边 W3/W4 RTN 与 GPTQ，这里 Q8_0/Q4_1/Q4_0 块格式 |
| [llama-cpp-7b](solutions/edge-runtime/llama-cpp-7b/README.md) | [15-849 编译与压缩的 Qwen2.5-7B 一节](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/README.md)、[15-849 服务与显存](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/serving-memory/README.md) | 同一个模型、同一张 RTX 6000D。那边 W4 g128 GPTQ 困惑度 +2.9%、AWQ +3.8%；这里 Q4_K_M + imatrix +1.2%（7.1016 → 7.1846）。两边协议不同（HF ctx 1024 对 llama-perplexity ctx 512），只比相对涨幅；imatrix 和 AWQ / GPTQ 一样按激活给误差加权 |
| [tinyengine-memory](solutions/mit6.5940/tinyengine-memory/README.md) | [15-849 分布式训练](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/distributed-training/README.md) | 那边全量重计算 +18.9% 时间；这里 patch 推理 +7.7% 整网 MACs 换 2.66 倍峰值 |
| [dma-double-buffer](solutions/edge-runtime/dma-double-buffer/README.md) | [15-849 分布式训练](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/distributed-training/README.md) | 1F1B 气泡与双缓冲的填充-排空项用同一个事件模拟器思路验证 |

## 面试时怎么切换

- 被问「decode 为什么慢」：先在 mlsys-infra 的 H100 数字上讲带宽受限，再切到本仓库 Q4_0 GEMV 在本机打到的权重带宽，说明同一条 E 公式在两边给出的下限。实测佐证：llama.cpp 在 RTX 6000D 上 F16 解码到带宽线的 93%，单线程 CPU 解码到单核带宽的 93%。
- 被问「端侧和服务端选哪种 4-bit 格式」：GPU 上 K-quant 的解包被藏住，选 Q4_K_M；CPU 上 K-quant 受指令限，Q4_0 快 1.48 倍。并发 128 时格式差距缩到 1.14 倍，这时转到 mlsys-infra 的 continuous batching 和分页 KV。
- 被问「量化为什么掉点」：LLM 侧讲离群通道（mlsys-infra M17），CNN 侧讲 BN 折叠后的通道幅值差和 KL 的零值尖峰（本仓库 M02、M03），两者都是「范围被少数值决定」。
- 被问「内存不够怎么办」：训练侧 ZeRO 和重计算，推理端 arena 规划、原地 depthwise、patch 推理。都先给出下界再谈怎么逼近。
