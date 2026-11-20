# 课程合并设计

端侧部署面试把同一条模型问成四笔账：放不放得进 SRAM 或内存，整数化之后哪一层坏了，算子选 im2col、直接卷积还是 depthwise 专用核，分块之后 DMA 能不能被算力盖住。本仓库按这四笔账排成 8 个模块，不按学校课表排。

[mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/CURRICULUM.md) 用 HBM、GPU 和多卡通信回答「算得过来、放得下、传得过来」。那些公式在这里仍然成立，只是拐点的带宽从 3 TB/s 换成几十 GB/s 的 LPDDR，或者几百 MB/s 量级的 SRAM，累加器从 Tensor Core 换成 int32。

## 用到的公开材料

| 材料 | 版本 | 本仓库拿哪一段 | 入口 |
| --- | --- | --- | --- |
| MIT 6.5940 TinyML and Efficient Deep Learning Computing | 2024 秋，Song Han。2025 秋因学术休假未开 | 五个 lab 按官方题面做完，期末项目做 MCU 端到端部署；MCUNet、TinyEngine 两讲进模块 01、04。GPTQ 与更大模型上的剪枝、AWQ 对照在 mlsys-infra | https://hanlab.mit.edu/courses/2024-fall-65940 |
| Machine Learning Compilation | 2022 夏，Tianqi Chen。中文班同期 | 端到端执行、TensorIR、自动优化、计算图。作业在 `mlc-ai/notebooks` 的 `assignment/` | https://mlc.ai/summer22-zh/ |
| Harvard TinyML 与 CS249r | HarvardX 微专业；CS249r 2020/2022/2023 | 256 KB 级 SRAM 上的唤醒词 / 视觉唤醒词任务定义。板级例程不入库 | https://mlsysbook.ai/courses/tinyml/ |
| CMSIS-NN | 现行主线 | int8 卷积的对称权重、per-channel scale、int32 bias。当契约读，不链接进本仓库 | https://github.com/ARM-software/CMSIS-NN |
| TFLite Micro / LiteRT | 现行主线 | arena 分配器、算子扁平图。解释器不整体搬进 `solutions/` | https://github.com/tensorflow/tflite-micro |
| llama.cpp、llama2.c | 现行主线 | Q4_0 / Q8_0 的块布局。GPTQ/AWQ 的求解在 mlsys-infra 模块 17 | https://github.com/ggml-org/llama.cpp |
| PULP Dory、Deeploy | 论文与仓库现行主线 | 分块、DMA、双缓冲的代价模型 | https://github.com/pulp-platform/deeploy |
| ExecuTorch、ncnn、MNN、RKNN Model Zoo、TensorRT | 现行主线 | 导出、分区、不支持算子的回退。闭源 SDK 不入库 | 见 [resources](resources/README.md) |

6.5940 的五个 lab 是：剪枝、量化、神经架构搜索、LLM 压缩、在笔记本上部署 LLM。八个模块只取其中与端侧部署契约有关的部分；五个 lab、期末项目和 MLC 作业 1 另按官方题面独立做了一遍（[lab1-pruning](solutions/mit6.5940/lab1-pruning/README.md)、[lab2-vgg](solutions/mit6.5940/lab2-vgg/README.md)、[lab3-nas](solutions/mit6.5940/lab3-nas/README.md)、[project-mcu-e2e](solutions/mit6.5940/project-mcu-e2e/README.md)、[lab4-awq](solutions/mit6.5940/lab4-awq/README.md)、[lab5-tinychat](solutions/mit6.5940/lab5-tinychat/README.md)、[e2e-relax](solutions/mlc/e2e-relax/README.md)）。

## 8 个模块

| # | 模块 | 要回答的问题 | 主要材料 | 实验 |
| ---: | --- | --- | --- | --- |
| 01 | 端侧 roofline | 这台设备的带宽、峰值和拐点是多少，depthwise 和 GEMV 落在哪一侧 | TinyEngine 讲义；STREAM；设备手册 | [host-roofline](solutions/edge-runtime/host-roofline/README.md) |
| 02 | 整数部署契约 | requant 怎么逐位对齐 ORT，int8 GEMM 的速度由什么决定 | 6.5940 Lab 2；gemmlowp；CMSIS-NN；ONNX Runtime QLinearConv | [int8-kernels](solutions/edge-runtime/int8-kernels/README.md) |
| 03 | 校准与误差定位 | 四种校准器差多少，KL 为什么会失败，掉点后怎么定位、怎么补 | 6.5940 Lab 2；TensorRT 8-bit Inference；ORT quantization | [lab2-quantization](solutions/mit6.5940/lab2-quantization/README.md) |
| 04 | 静态内存规划 | 激活峰值的下界、规划器离下界多远、原地与 patch 能压多少 | TFLite Micro arena；MCUNet / MCUNetV2 | [tinyengine-memory](solutions/mit6.5940/tinyengine-memory/README.md) |
| 05 | 卷积实现 | im2col、implicit GEMM、depthwise、Winograd 各在哪个区间赢 | ncnn / CMSIS-NN 的内核分工；Lavin & Gray 2016 | [conv-kernels](solutions/edge-runtime/conv-kernels/README.md) |
| 06 | 调度搜索 | 调度空间多大、什么样的调度快、代价模型和测量噪声能信多少 | MLC 自动程序优化；AutoTVM；Ansor | [tensor-program-tuning](solutions/mlc/tensor-program-tuning/README.md) |
| 07 | DMA 与双缓冲 | 什么时候重叠省时间，什么时候省不了，分块怎么选 | Dory、Deeploy、TinyEngine | [dma-double-buffer](solutions/edge-runtime/dma-double-buffer/README.md) |
| 08 | 块量化解码 | Q4_0 的字节、解码上界、内核能拿到多少、困惑度掉多少 | llama.cpp / ggml；GPTQ、AWQ 的对照在 mlsys-infra | [q4-gemv](solutions/edge-runtime/q4-gemv/README.md)、[llama-cpp-7b](solutions/edge-runtime/llama-cpp-7b/README.md) |

模块 08 有两个实验。q4-gemv 自己实现块格式和 GEMV，用「层时间之和」预测整模型解码；llama-cpp-7b 用 llama.cpp 跑完整的 Qwen2.5-7B，检验这个预测，并补上单层实验测不到的东西：11 种 GGUF 的质量代价、K-quant 在 CPU 与 GPU 上相反的速度排序、batch > 1 的吞吐。

模块 01、02、05、06、07、08 的实验还在 Xeon 8470Q + RTX 6000D 上重跑过一遍，用来区分「结构性」结论（只由访存 / 计算比决定）和「位置性」结论（取决于两个量谁大，换机器会翻转），见 [CROSS-HOST.md](interview/CROSS-HOST.md) 和[服务器复测报告](reports/edge-runtime-server/edge_runtime_server_report.pdf)。

逐道官方作业的分工见 [assignments/README.md](assignments/README.md)。

## 和相邻仓库的边界

| 内容 | 放在哪 | 理由 |
| --- | --- | --- |
| H100/A100 roofline、FlashAttention、CUDA graph | mlsys-infra | 带宽差一个数量级，面试时两套数不要混 |
| GPTQ、AWQ、SmoothQuant、LLM.int8 的求解 | mlsys-infra 模块 17 | 这里只消费它们产出的权重布局 |
| KV cache、推测解码的正确性和 GPU 加速比 | mlsys-infra 模块 15 | 模块 08 只补 LPDDR 上的权重字节和块缩放 |
| 动作块的同步执行和异步执行 | physical-ai | 那是控制周期。模块 07 是计算和 DMA 的重叠 |
| 厂商板级 SDK、带账号的工具链 | 不入库 | 结果可以记在本地笔记，仓库里只留可复现的 CPU 路径 |

## 每个模块交付什么

1. 一篇笔记，放在 `notes/`，16 节结构，至少 5 层追问链，并链到 mlsys-infra 的对应模块（`notes/check_notes.py` 校验）。
2. 公式进 [interview/FORMULAS.md](interview/FORMULAS.md)（E1–E27），在实验代码或 `tools/edge_calc.py` 里有测试。
3. 一个实验：`README.md`、`tests/`、`scripts/run_experiments.py`、`artifacts/raw/`、`artifacts/figures/`；README 写明和原课程的差异与「不能推出什么」。
4. 笔记和 README 里引用的关键数字进 [`claims.json`](claims.json)，由 `scripts/check_claims.py` 对原始结果校验。
5. 至少一道带常见错答的题进 [interview/QUESTION-BANK.md](interview/QUESTION-BANK.md)，一句话答案进 [COURSE-QA](interview/COURSE-QA.md)。
6. 一份 LaTeX 报告，放在 `reports/<lab>/`，带参考文献（`scripts/check_reports.py` 校验）。

八个模块六件都已交付，第 9 个实验 llama-cpp-7b、第 10 个实验 runtime-compare 和五个官方作业也有 README、测试、claims 和报告，状态见 [PROGRESS.md](PROGRESS.md)。

## 岗位怎么读

| 岗位 | 先做 | 开口用 |
| --- | --- | --- |
| 模型部署 / 量化 | 02、03 | KL 校准在 ReLU6 网络上掉 24 个点；按敏感度混合精度 +9.5 KiB 回到 87.69% |
| MCU / CMSIS-NN | 01、02、04、07 | int32 累加与 requant 逐位对齐；激活峰值下界与 patch 推理 |
| NPU 编译 / mapping | 04、06、07 | 寄存器分块、代价模型 Spearman 0.47–0.53；depthwise 在片外存储上 DMA 受限 |
| CPU 推理引擎（ncnn、MNN、XNNPACK） | 02、05、06 | int8 GEMM 的打包与融合 requant（30.8 倍、3.3 倍）；depthwise NHWC 8–9 倍 |
| 端侧 LLM | 08，外加 mlsys-infra 的 M15 和 M17 | tok/s ≤ 带宽 / 权重字节；Q4 内核只拿到字节比的 65%；llama.cpp 跑 7B 时 CPU 上 K-quant 只到带宽线的 44–56%，GPU 上仍有 75% |
