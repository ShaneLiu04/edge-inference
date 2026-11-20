# Kiloscale

**Edge Inference** · 端侧部署与推理优化 · *From kernels to kilobytes*

<p align="center">
  <a href="https://github.com/ShaneLiu04/edge-inference/actions/workflows/ci.yml"><img src="https://github.com/ShaneLiu04/edge-inference/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-3da639?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+">
  <a href="VERIFICATION.md"><img src="https://img.shields.io/badge/claims-227-8a6d3b?style=flat-square" alt="227 claims"></a>
</p>

端侧部署的面试通常围绕两个问题：单层运行耗时，以及误差从哪一层引入。本仓库将其组织为 8 个模块、10 个实验和 7 套官方复现；所有数字均可回溯至原始结果。

<p align="center">
  <img src="assets/kiloscale-cover.png" alt="Kiloscale · Edge Inference · 面向面试的端侧部署与推理优化" width="100%">
</p>

改变尺度、布局、分块或数据格式后，相关数字会随之变化。KL 校准失效、Winograd 跨机器性能反转、双缓冲在 CPU 上性能下降，上述结果均保留在仓库中，并在面试材料中给出解释。

[三项实测结果](#三项实测结果) · [课程与材料](#课程与材料) · [八个模块](#八个模块) · [官方作业与期末项目](#官方作业与期末项目) · [面试准备](#面试准备) · [快速开始](#快速开始) · [实验报告](#实验报告) · [仓库结构](#仓库结构)

## 三项实测结果

**布局决定性能。** 使用相同的 VNNI 指令时，MobileNetV2 的 pw_expand_56（K = 24）上按点积实现的内核为 4.2 GOP/s；将 B 预打包为 VNNI 所需的 4 字节组后达到 129 GOP/s（30.8 倍）。接入标量 requant 后降至 20.2 GOP/s；将其融合进 SIMD 尾段后恢复至 66.0 GOP/s（3.3 倍）。K 越短，尾段占比越高。

![int8 GEMM](solutions/edge-runtime/int8-kernels/artifacts/figures/gemm_bench.png)

**校准不应将零纳入分布拟合。** TensorRT 式 KL 校准在 ReLU6 网络上将激活范围裁至 MSE 最优值的约四分之一（红柱，中位 0.268），W8A8 精度从 88% 降至 63.68%。原因是 ReLU 输出中包含大量精确的零，被误作待拟合分布；剔除这些零后（蓝柱）恢复至 88.06%。

![KL](solutions/mit6.5940/lab2-quantization/artifacts/figures/kl_points.png)

**以重算降低峰值内存。** MobileNetV2-1.0@224 前 4 个 block 采用 2×2 patch 推理，int8 激活峰值从 1.44 MiB 降至 554 KiB，整网 MACs 仅增加 7.7%；扩展至 6 个 block、4×4 时 MACs 增加 59%，方可放入 STM32F746 的 320 KB SRAM。

![patch](solutions/mit6.5940/tinyengine-memory/artifacts/figures/patch_tradeoff.png)

## 课程与材料

| 课程 / 材料 | 用途 |
| --- | --- |
| [MIT 6.5940 · TinyML and Efficient Deep Learning](https://hanlab.mit.edu/courses/2024-fall-65940)（Song Han，Fall 2024） | 8 个模块的主要依据；Lab 1–5：VGG 剪枝与量化、OFA-MCUNet 上的 NAS、OPT-1.3B 的 AWQ、TinyChat 的 W4A8 与 LLaMA2-7B。 |
| [Machine Learning Compilation](https://mlc.ai/)（Tianqi Chen；[MLC notebooks](https://github.com/mlc-ai/notebooks)） | MLC 作业 1：TVM 0.27 的 Relax 与 MetaSchedule；调度搜索参考 TensorIR / Ansor 的思路。 |
| [MCUNet / MCUNetV2](https://github.com/mit-han-lab/mcunet)（MIT HAN Lab） | Lab 3 的 OFA 超网与 VWW 数据（课程分发的 mcunetv2-dev）；期末项目将 MCUNet 官方 VWW 模型纳入同一张对照表。 |
| [TinyEngine](https://github.com/mit-han-lab/tinyengine)（MIT HAN Lab） | 静态内存规划、patch 推理与 MCU 延迟拟合；M04 的实验依据。 |
| [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) · [TFLite Micro](https://github.com/tensorflow/tflite-micro) · [ONNX Runtime](https://onnxruntime.ai/) · [PULP Dory](https://github.com/pulp-platform/dory) · [llama.cpp](https://github.com/ggml-org/llama.cpp) | int8 契约、六条 CPU 推理路径、DMA 双缓冲、Q4 解码与 7B 端到端。 |
| [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/) · [Harvard TinyML](https://mlsysbook.ai/courses/tinyml/) | 256 KB SRAM 的任务口径与精度 / 延迟 / Flash / 峰值 SRAM 四列；仅采用任务口径，不构成 MLPerf 提交，也不训练 KWS。 |

官方课程、论文与仓库的版权归原作者；题面与 starter 的来源见 [UPSTREAM.md](UPSTREAM.md)，本仓库不镜像官方 notebook 或 starter。模块与讲次的对应见 [CURRICULUM.md](CURRICULUM.md)。

## 八个模块

| 模块 | 核心结论 | 笔记 | 实验 |
| --- | --- | --- | --- |
| 端侧 roofline | 带宽决定上限。单线程 DRAM 13.4 GB/s；fp32 峰值 155 GFLOP/s（ridge 11.5），int8 396 GOP/s（ridge 29.5）。depthwise 3×3 为 4.5 FLOP/B，始终位于内存墙左侧。 | [M01](notes/module-01-edge-roofline.md) | [host-roofline](solutions/edge-runtime/host-roofline/README.md) |
| 整数契约 | `real = s·(q − z)`；累加器单位为 s_w·s_x。gemmlowp 的两步定点与 ORT 的浮点 round-half-even 在平局上不同：构造平局时 1.03% 的输出相差 1。 | [M02](notes/module-02-integer-contract.md) | [int8-kernels](solutions/edge-runtime/int8-kernels/README.md) |
| 校准与定位 | 权重与激活分别分析，再进行逐层敏感度与 SQNR 评估。W4 逐张量 76.42、逐通道 83.20；按敏感度提升 8 层至 W8 仅增加 9.5 KiB，精度恢复至 87.69；按参数量提升 8 层增加 78 KiB，精度为 83.62。 | [M03](notes/module-03-calibration-localization.md) | [lab2-quantization](solutions/mit6.5940/lab2-quantization/README.md) |
| 静态内存 | 同时存活张量的大小之和构成下界。按大小降序贪心在 11 个配置上均等于下界；first-fit 多 13.3%，原地 depthwise 再降低 6.7%。 | [M04](notes/module-04-static-memory-planning.md) | [tinyengine-memory](solutions/mit6.5940/tinyengine-memory/README.md) |
| 卷积 | im2col 工作区为 patch 的 9 倍；implicit GEMM 不需要工作区，在 stem 上快 1.8 倍。depthwise 采用 NHWC 才能沿通道向量化（8–9 倍）；Winograd F(2,3) 在 14×14 上为 2.15 倍，在 56×56 上仅 1.16 倍。 | [M05](notes/module-05-convolution-kernels.md) | [conv-kernels](solutions/edge-runtime/conv-kernels/README.md) |
| 调度搜索 | 有效调度以寄存器块为单位，而非 L1 块：最内层 2–4 行，16 个 ymm 累加器。进化搜索 84.5 GFLOP/s，教科书分块 35.4；同一调度复测差异为 −9%～+5%，前几名无法区分。 | [M06](notes/module-06-schedule-search.md) | [tensor-program-tuning](solutions/mlc/tensor-program-tuning/README.md) |
| 双缓冲 | `T = n·max(m, c) + min(m, c)`，优化的是较短的一侧。HyperRAM 上的 depthwise 受 DMA 限制，效率 0.257，双缓冲仅提升 1.24 倍；本机以两线程模拟 4 KiB tile 时性能下降（0.61 倍）。 | [M07](notes/module-07-dma-double-buffering.md) | [dma-double-buffer](solutions/edge-runtime/dma-double-buffer/README.md) |
| 端侧 LLM | tok/s ≤ 带宽 / 权重字节。Q4_0 为 4.5 bit/权重，7B 为 3.79 GB，Pi 5 上限为 4.5 tok/s。本机 Q4 × Q8 VNNI 比 fp32 快 4.6 倍（字节少 7.1 倍），其余开销来自每个块的解包指令；GPT-2 Q4_0 困惑度从 29.95 升至 31.88。 | [M08](notes/module-08-block-quantized-decode.md) | [q4-gemv](solutions/edge-runtime/q4-gemv/README.md) |

另外两个实验分别使用 llama.cpp 运行完整的 Qwen2.5-7B（11 种 GGUF），评估带宽线、KLD 与端到端常数；runtime-compare 将 lab2 的同一张网络交由 PyTorch、ONNX Runtime fp32/int8、LiteRT、ExecuTorch、TVM 六条路径，比较精度与 CPU 时间。说明见 [llama-cpp-7b](solutions/edge-runtime/llama-cpp-7b/README.md) 与 [runtime-compare](solutions/edge-runtime/runtime-compare/README.md)。

简要答案汇总见 [课程问答](interview/COURSE-QA.md)，常见错答见 [题库](interview/QUESTION-BANK.md)。

## 官方作业与期末项目

题面见 [assignments/](assignments/README.md)；本仓库提供个人实现，不镜像 starter，来源见 [UPSTREAM.md](UPSTREAM.md)。

| 作业 | 结果 | 说明 |
| --- | --- | --- |
| 6.5940 Lab 1 剪枝 | 细粒度剪枝至 23.6% 尺寸，微调后 92.72%（稠密 92.95%）；通道剪枝 30%：MACs 0.504 倍、延迟 0.64 倍、92.52%。 | [lab1-pruning](solutions/mit6.5940/lab1-pruning/README.md) |
| 6.5940 Lab 2 量化 | k-means 8 / 4 / 2 bit 经 QAT 后分别为 92.82 / 92.84 / 92.18%；整数 int8 VGG 92.94%，int4 PTQ 89.98%。 | [lab2-vgg](solutions/mit6.5940/lab2-vgg/README.md) |
| 6.5940 Lab 3 NAS | 闭式 MACs / 峰值在 1004 个子网上与课程计数器一致（差异为 0），速度快 310 倍；预测器 MAE 0.18 pp、Kendall 0.925；250 KB / 60M 实测 92.556%，200 KB / 30M 90.372%。 | [lab3-nas](solutions/mit6.5940/lab3-nas/README.md) |
| 6.5940 期末项目 | Lab 3 子网 int8 92.53%，比 mcunet-vww2 高 1.17 pp，估算延迟为 223 ms 与 234 ms；patch 后 SRAM 从 250 KB 降至 122 KB。 | [project-mcu-e2e](solutions/mit6.5940/project-mcu-e2e/README.md) |
| 6.5940 Lab 4 AWQ | OPT-1.3B W3 g128：RTN 107.97，按激活保留 1% 16.69、按权重 103.41；AWQ 16.87，加入截断后 16.04（fp16 14.47）。 | [lab4-awq](solutions/mit6.5940/lab4-awq/README.md) |
| 6.5940 Lab 5 TinyChat | W4A8 内核全部通过；all_techniques 为参考实现的 28.2 倍，LLaMA2-7B decode 为 12.5 倍；在 `-O3` 下为 51.5 倍、191 ms/token。 | [lab5-tinychat](solutions/mit6.5940/lab5-tinychat/README.md) |
| MLC 作业 1 | TVM 0.27 Relax 五种写法均为 83.88%；默认 2375 µs（linear0 占 2344）→ 手写调度 157 → MetaSchedule 44，PyTorch 505。 | [e2e-relax](solutions/mlc/e2e-relax/README.md) |

## 面试准备

90 分钟内可准备的内容：熟记 [硬件速查表](interview/HARDWARE-NUMBERS.md)，默写公式卡 E1、E4、E10、E15、E25，并准备一个 [作品集案例](interview/PORTFOLIO.md)。按岗位选择案例：部署与量化岗位讲解案例 1（KL 校准失效）和 3（整数逐位对齐）；推理引擎岗位讲解案例 2（int8 GEMM 布局）和 6（Q4 解码）；NPU 编译岗位讲解案例 4（内存规划）和 5（调度搜索）。

周末阅读顺序：按八个模块从上到下，每行阅读后查阅笔记中的「白板估算」和「面试追问链」。手写题见 [CODING.md](interview/CODING.md)，排查顺序见 [ENGINEERING.md](interview/ENGINEERING.md)，设计题见 [SYSTEM-DESIGN.md](interview/SYSTEM-DESIGN.md)。按岗位的阅读路线见 [READING-ROADMAP](notes/READING-ROADMAP.md)。

估算可手算，也可用计算器复核：

```bash
python tools/edge_calc.py gemv --n 4096 --k 4096 --dtype int8
python tools/edge_calc.py qblock --params 6.74e9 --dtype q4_0 --bw 17.068e9
python tools/edge_calc.py pipeline --tiles 8 --dma 2 --compute 2
```

## 快速开始

```bash
git clone https://github.com/ShaneLiu04/edge-inference.git
cd edge-inference
pip install -r requirements.txt
```

依赖包括 CPU torch、onnxruntime、numpy，以及 PATH 中的 g++。q4-gemv 的 Triton 部分见 `requirements-gpu.txt`；llama-cpp-7b 需要一份编译好的 llama.cpp（测试仅覆盖解析器）；e2e-relax 需要单独安装 TVM 0.27 的 venv，未安装时跳过 TVM 测试。各实验的运行方式：

```bash
cd solutions/<course>/<lab>
python -m pytest -q tests
python scripts/run_experiments.py
```

全部测试：`./scripts/run-all-tests.ps1`（Linux：`scripts/run-all-tests.sh`；18 套 206 个测试，未安装 TVM 时跳过 3 个；`EDGE_MARCH=haswell` 验证无 AVX-512 的回退路径）。数字复核：`python scripts/check_claims.py`。笔记结构：`python notes/check_notes.py`。仓库结构、链接、报告与密钥检查：`./scripts/check-repo.ps1`。以上检查均在 [CI](.github/workflows/ci.yml) 中执行，结果记录于 [VERIFICATION.md](VERIFICATION.md)。

## 实验报告

每个实验对应一份中文报告（LaTeX / PDF），结构一致：实验目标、逐图解读、实测与模型对照、面试追问、20 秒讲法、结论边界。以下 6 份为优先阅读材料：

| 报告 | 关键结论 |
| --- | --- |
| [host-roofline](reports/host-roofline/host_roofline_report.pdf) | 单核只读 13.4 GB/s 是在途请求数的上限，不是总线带宽；int8 峰值是 fp32 的 2.56 倍，不是 4 倍。 |
| [edge-runtime-server](reports/edge-runtime-server/edge_runtime_server_report.pdf) | 同一份代码在 Xeon 8470Q / RTX 6000D 上重跑，大部分指标仅改变常数，三条经验法则发生反转。 |
| [llama-cpp-7b](reports/llama-cpp-7b/llama_cpp_7b_report.pdf) | 只计算 Linear 的预测与 llama.cpp 端到端结果之间相差一个 0.41–0.47 ms/token 的常数；CPU 上 Q8_0 最快，GPU 上 Q4_K_M 最快。 |
| [runtime-compare](reports/runtime-compare/runtime_compare_report.pdf) | 单线程 ORT int8 127 µs、LiteRT 166、ExecuTorch 192、ORT fp32 202、PyTorch eager 1173、未调优 TVM 4107。 |
| [mit6.5940-labs](reports/mit6.5940-labs/mit65940_labs_report.pdf) | 显著性来自激活而非权重（保留 1%：16.69 对 103.41）；NAS 在 0.4 pp 的验证噪声下只能选出一组较优子网，无法稳定排名。 |
| [project-mcu-e2e](reports/project-mcu-e2e/project_mcu_e2e_report.pdf) | 搜出的子网在同等估算延迟下比 MCUNet 官方模型高 1.2 个点，但 Flash 多 190 KB；patch 对不同网络的收益差异较大。 |

其余单实验报告位于 [`reports/`](reports/) 下，包括 int8-kernels、conv-kernels、dma-double-buffer、tensor-program-tuning、q4-gemv、lab2-quantization、tinyengine-memory、mlc-e2e-relax。

## 仓库结构

- 课程合并方式与模块映射：[CURRICULUM.md](CURRICULUM.md)
- 公式、设计题、案例、手写题、错答：[`interview/`](interview/README.md)
- 单讲推导、白板估算与追问链：[`notes/`](notes/README.md)，从 [M01](notes/module-01-edge-roofline.md) 到 [M08](notes/module-08-block-quantized-decode.md)
- 实验代码、测试、图与结论：[`solutions/`](solutions/README.md)
- 数字来源与复核方式：[VERIFICATION.md](VERIFICATION.md)、[`claims.json`](claims.json)
- 官方题面与 starter：[`assignments/`](assignments/README.md)，来源见 [UPSTREAM.md](UPSTREAM.md)
- 两台主机的同项对照：[CROSS-HOST.md](interview/CROSS-HOST.md)
- GPU 侧姊妹仓库 mlsys-infra 的对应：[BRIDGE.md](BRIDGE.md)
- 论文、运行时与工具索引：[`resources/`](resources/README.md)

## 做法与边界

实验最初在 4 物理核 Ice Lake 虚拟机（AVX-512 + VNNI，Windows，g++ 13.2）上完成，不需要 GPU 或开发板；同一份代码随后迁移至 AutoDL 的 2 × Xeon Platinum 8470Q（Sapphire Rapids，AMX）和 RTX 6000D 上重跑，检查哪些结论在更换机器后发生反转。另外两个实验补充 llama.cpp 运行完整的 Qwen2.5-7B，以及同一张网络走 6 条 CPU 推理路径的横评。官方部分依据题面独立完成 MIT 6.5940 的 Lab 1–5 和 MLC 作业 1；期末项目将 Lab 3 搜出的子网与 MCUNet 官方 VWW 模型纳入同一张表。

作者 [ShaneLiu04](https://github.com/ShaneLiu04)，与上述学校、实验室和项目无隶属关系。`solutions/` 为个人实现，正在修读相关课程的同学请勿直接将其作为作业提交。

## License

官方材料的版权归原作者，来源见 [UPSTREAM.md](UPSTREAM.md)。笔记、面试材料、工具、脚本与实验代码使用 [MIT License](LICENSE)。
