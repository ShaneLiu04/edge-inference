# 端侧部署与推理优化

面向端侧推理、模型部署和 NPU 编译岗位的课程仓库。八讲笔记、公式卡和八个实验只为一件事服务：当场把一层卷积或一次解码的账算完，说出误差从哪一层进来，并说明换一种尺度、布局、分块或格式之后数字怎么变。

材料来自 MIT 6.5940 TinyML and Efficient Deep Learning（Song Han，Fall 2024）、Tianqi Chen 的 Machine Learning Compilation，以及 CMSIS-NN、TFLite Micro、ONNX Runtime、PULP Dory、llama.cpp 这些公开实现里反复出现的契约。取舍见 [CURRICULUM.md](CURRICULUM.md)。作者 [ShaneLiu04](https://github.com/ShaneLiu04)，与上述学校、实验室和项目无隶属关系。`solutions/` 是个人实现，正在修这些课的同学不要把它当作作业提交。

全部实验在一台 4 物理核的 Ice Lake 服务器虚拟机（AVX-512 + VNNI，Windows，g++ 13.2）上完成，不需要 GPU 和开发板；每个数字都能由 `python scripts/check_claims.py` 对照原始结果复核。

数据中心一侧（GPU、FlashAttention、GPTQ / AWQ、KV cache、分布式训练）在姊妹仓库 [mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra)。两边按同一套结构写（16 节笔记、claims、VERIFICATION、作品集），对应关系见 [BRIDGE.md](BRIDGE.md)。

## 三张实测图

int8 GEMM 的速度由布局和尾段决定，不由指令决定。同样用 VNNI，按点积写的内核在 MobileNetV2 的 pw_expand_56（K = 24）上只有 4.2 GOP/s，把 B 预先打包成 VNNI 需要的 4 字节组后是 129 GOP/s（30.8 倍）；
再接一个标量 requant 就掉回 20.2，融合进 SIMD 尾段回到 66.0（3.3 倍）。K 越短，尾段占比越大。

![int8 GEMM](solutions/edge-runtime/int8-kernels/artifacts/figures/gemm_bench.png)

TensorRT 式 KL 校准在 ReLU6 网络上把激活范围裁到 MSE 最优值的四分之一左右（红柱，中位 0.268），W8A8 精度从 88% 掉到 63.68%。原因是 ReLU 输出里大量精确的零被当成要拟合的分布；去掉零之后（蓝柱）回到 88.06%。

![KL](solutions/mit6.5940/lab2-quantization/artifacts/figures/kl_points.png)

按 patch 推理用重算换峰值内存。MobileNetV2-1.0@224 前 4 个 block 切 2×2，int8 激活峰值从 1.44 MiB 降到 554 KiB，整网 MACs 只多 7.7%；切到 6 个 block、4×4 时 MACs 多 59%，才勉强放进 STM32F746 的 320 KB。

![patch](solutions/mit6.5940/tinyengine-memory/artifacts/figures/patch_tradeoff.png)

## 开口的一句

往下的推导在笔记，可手推的步骤在 [公式卡](interview/FORMULAS.md)（E1–E27），90 秒的讲法在 [作品集](interview/PORTFOLIO.md)，硬件数字统一查 [速查表](interview/HARDWARE-NUMBERS.md)。

| 模块 | 开口 | 笔记 | 证据 |
| --- | --- | --- | --- |
| 端侧 roofline | 先写带宽。本机单线程 DRAM 13.4 GB/s、fp32 峰值 155 GFLOP/s，ridge 11.5；int8 峰值 396 GOP/s，ridge 29.5。depthwise 3×3 只有 4.5 FLOP/B，永远在左边。 | [M01](notes/module-01-edge-roofline.md) | [host-roofline](solutions/edge-runtime/host-roofline/README.md) |
| 整数契约 | `real = s·(q − z)`；累加器单位是 s_w·s_x。gemmlowp 两步定点和 ORT 的浮点 round-half-even 在平局上分道扬镳：构造平局时 1.03% 的输出差 1。 | [M02](notes/module-02-integer-contract.md) | [int8-kernels](solutions/edge-runtime/int8-kernels/README.md) |
| 校准与定位 | 先分开权重与激活，再逐层敏感度，再看 SQNR。W4 逐张量 76.42、逐通道 83.20；按敏感度升 8 层到 W8 只多 9.5 KiB 就回到 87.69，按参数量升 8 层多 78 KiB 只有 83.62。 | [M03](notes/module-03-calibration-localization.md) | [lab2-quantization](solutions/mit6.5940/lab2-quantization/README.md) |
| 静态内存 | 同时存活之和是下界。按大小降序贪心在 11 个配置上全部等于下界，first-fit 多 13.3%；原地 depthwise 再省 6.7%。 | [M04](notes/module-04-static-memory-planning.md) | [tinyengine-memory](solutions/mit6.5940/tinyengine-memory/README.md) |
| 卷积 | im2col 工作区是 9 倍 patch；implicit GEMM 不要工作区，stem 上快 1.8 倍。depthwise 换成 NHWC 才能沿通道向量化（8–9 倍）；Winograd F(2,3) 在 14×14 上 2.15 倍，56×56 上只有 1.16 倍。 | [M05](notes/module-05-convolution-kernels.md) | [conv-kernels](solutions/edge-runtime/conv-kernels/README.md) |
| 调度搜索 | 好调度是寄存器块，不是 L1 块：最内层 2–4 行，16 个 ymm 累加器。进化搜索 84.5 GFLOP/s，教科书分块 35.4；但同一调度复测差 −9%～+5%，前几名不可区分。 | [M06](notes/module-06-schedule-search.md) | [tensor-program-tuning](solutions/mlc/tensor-program-tuning/README.md) |
| 双缓冲 | `T = n·max(m, c) + min(m, c)`，省的是短的一侧。HyperRAM 上的 depthwise 是 DMA 受限，效率 0.257，双缓冲只快 1.24 倍；本机两线程模拟 4 KiB tile 时反而慢（0.61 倍）。 | [M07](notes/module-07-dma-double-buffering.md) | [dma-double-buffer](solutions/edge-runtime/dma-double-buffer/README.md) |
| 端侧 LLM | tok/s ≤ 带宽 / 权重字节。Q4_0 4.5 bit/权重，7B 3.79 GB，Pi 5 上限 4.5 tok/s。本机 Q4 × Q8 VNNI 比 fp32 快 4.6 倍（字节少 7.1 倍），剩下的被每块的解包指令吃掉；GPT-2 Q4_0 困惑度 29.95 → 31.88。 | [M08](notes/module-08-block-quantized-decode.md) | [q4-gemv](solutions/edge-runtime/q4-gemv/README.md) |

一句话答案的全表在 [课程问答](interview/COURSE-QA.md)，带常见错答的题在 [题库](interview/QUESTION-BANK.md)。

## 怎么准备

明天就面试：背 [速查表](interview/HARDWARE-NUMBERS.md) 的本机实测与设备表，默写公式卡的 E1、E4、E10、E15、E25，准备一个 [作品集故事](interview/PORTFOLIO.md)：部署 / 量化岗讲 1（KL 校准失败）和 3（整数逐位对齐），推理引擎岗讲 2（int8 GEMM 布局）和 6（Q4 解码），NPU 编译岗讲 4（内存规划）和 5（调度搜索）。

一个周末：按上表从上到下，每行先讲开口的那句，再打开笔记的「白板估算」和「面试追问链」。手写题在 [CODING.md](interview/CODING.md)，排查顺序在 [ENGINEERING.md](interview/ENGINEERING.md)，设计题在 [SYSTEM-DESIGN.md](interview/SYSTEM-DESIGN.md)。

按岗位的阅读路线在 [READING-ROADMAP](notes/READING-ROADMAP.md)。估算可以用计算器复核：

```bash
python tools/edge_calc.py gemv --n 4096 --k 4096 --dtype int8
python tools/edge_calc.py qblock --params 6.74e9 --dtype q4_0 --bw 17.068e9
python tools/edge_calc.py pipeline --tiles 8 --dma 2 --compute 2
```

## 材料放在哪

| 要找什么 | 打开 |
| --- | --- |
| 课程怎么合并、每个模块对应哪几讲 | [CURRICULUM.md](CURRICULUM.md) |
| 公式、设计题、故事、手写题、错答 | [`interview/`](interview/README.md) |
| 一讲的推导、白板估算和追问链 | [`notes/`](notes/README.md)，从 [M01](notes/module-01-edge-roofline.md) 到 [M08](notes/module-08-block-quantized-decode.md) |
| 某个实验的代码、测试、图和结论 | [按实验查](#实验实现)，总目录在 [`solutions/`](solutions/README.md) |
| 数字从哪来、怎么复核 | [VERIFICATION.md](VERIFICATION.md)、[`claims.json`](claims.json) |
| 官方题面和 starter | [`assignments/`](assignments/README.md)，来源记在 [UPSTREAM.md](UPSTREAM.md) |
| 与 mlsys-infra 的对应 | [BRIDGE.md](BRIDGE.md) |
| 论文、运行时、工具索引 | [`resources/`](resources/README.md) |

## 实验实现

每个 README 写明了和原课程的差异，以及这些数字**不能**推出什么。C++ 内核共用 [`edge_harness/include/edgebench.h`](edge_harness/include/edgebench.h)（计时、线程池、JSON 输出），Python 侧共用 [`edge_harness/`](edge_harness/)（编译缓存、结果写入、claims 校验）。

| 实验 | 代码、测试、图、说明 | 能引用的结果 |
| --- | --- | --- |
| host-roofline | [代码](solutions/edge-runtime/host-roofline/cpp/) · [测试](solutions/edge-runtime/host-roofline/tests/) · [图](solutions/edge-runtime/host-roofline/artifacts/figures/) · [说明](solutions/edge-runtime/host-roofline/README.md) | 1 / 4 线程 DRAM 13.4 / 48.4 GB/s，fp32 155 / 597 GFLOP/s，int8 396 / 1493 GOP/s；L1 → DRAM 带宽掉 12 倍 |
| int8-kernels | [代码](solutions/edge-runtime/int8-kernels/cpp/) · [测试](solutions/edge-runtime/int8-kernels/tests/) · [图](solutions/edge-runtime/int8-kernels/artifacts/figures/) · [说明](solutions/edge-runtime/int8-kernels/README.md) | 打包 VNNI 212–221 GOP/s，对点积版 2.0–30.8 倍；融合 requant 3.3 倍；浮点 requant 与 ORT QLinearConv 100% 逐位一致 |
| conv-kernels | [代码](solutions/edge-runtime/conv-kernels/cpp/) · [测试](solutions/edge-runtime/conv-kernels/tests/) · [图](solutions/edge-runtime/conv-kernels/artifacts/figures/) · [说明](solutions/edge-runtime/conv-kernels/README.md) | implicit GEMM 22.4–34.9 GFLOP/s 无工作区；depthwise NHWC 8.2–9.0 倍；Winograd 1.16–2.15 倍，F(4,3) 在 fp16 下误差是直接卷积的 7 倍 |
| lab2-quantization | [代码](solutions/mit6.5940/lab2-quantization/code/) · [测试](solutions/mit6.5940/lab2-quantization/tests/) · [图](solutions/mit6.5940/lab2-quantization/artifacts/figures/) · [说明](solutions/mit6.5940/lab2-quantization/README.md) | fp32 88.16%；W8A8 87.97–88.17%，KL 63.68%；混合精度 +9.5 KiB 回到 87.69%；QAT 87.63%；ORT int8 1.47 倍 |
| tinyengine-memory | [代码](solutions/mit6.5940/tinyengine-memory/code/) · [测试](solutions/mit6.5940/tinyengine-memory/tests/) · [图](solutions/mit6.5940/tinyengine-memory/artifacts/figures/) · [说明](solutions/mit6.5940/tinyengine-memory/README.md) | 贪心等于下界，first-fit +13.3%；原地 depthwise −6.7% / −30.8%；patch 2×2 峰值 −62%，MACs +7.7% |
| tensor-program-tuning | [代码](solutions/mlc/tensor-program-tuning/code/) · [测试](solutions/mlc/tensor-program-tuning/tests/) · [图](solutions/mlc/tensor-program-tuning/artifacts/figures/) · [说明](solutions/mlc/tensor-program-tuning/README.md) | 搜索 2.4 / 1.6 倍于教科书；代价模型 Spearman 0.47 / 0.53；复测噪声 −9%～+5%；512 位 13/16 变慢 |
| dma-double-buffer | [代码](solutions/edge-runtime/dma-double-buffer/code/) · [测试](solutions/edge-runtime/dma-double-buffer/tests/) · [图](solutions/edge-runtime/dma-double-buffer/artifacts/figures/) · [说明](solutions/edge-runtime/dma-double-buffer/README.md) | 模拟与闭式逐点相等；depthwise + HyperRAM 效率 0.257、双缓冲 1.24 倍；本机实测 0.61–1.23 倍 |
| q4-gemv | [代码](solutions/edge-runtime/q4-gemv/cpp/) · [测试](solutions/edge-runtime/q4-gemv/tests/) · [图](solutions/edge-runtime/q4-gemv/artifacts/figures/) · [说明](solutions/edge-runtime/q4-gemv/README.md) | 与 gguf 逐字节一致；Q4 × Q8 单线程 4.6 倍于 fp32；GPT-2 280 窗口 fp32 29.95 / Q8_0 29.95 / Q4_0 31.88 / Q4_1 32.15 |

依赖：`pip install -r requirements.txt`（CPU torch、onnxruntime、numpy）和 PATH 上的 g++。每个实验：

```bash
cd solutions/<course>/<lab>
python -m pytest -q tests
python scripts/run_experiments.py
```

全部测试：`./scripts/run-all-tests.ps1`（Linux：`scripts/run-all-tests.sh`；9 套 130 个测试，`EDGE_MARCH=haswell` 时验证无 AVX-512 的回退路径）。笔记结构：`python notes/check_notes.py`。数字复核：`python scripts/check_claims.py`。仓库结构、链接与密钥检查：`./scripts/check-repo.ps1`。都写进了 [CI 配置](.github/workflows/ci.yml)。

## 许可

官方材料的版权归原作者，来源记在 [UPSTREAM.md](UPSTREAM.md)。笔记、面试材料、工具、脚本和实验代码使用 [MIT License](LICENSE)。
