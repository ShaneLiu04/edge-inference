# 官方作业与对照项目

这里不放 starter，也不转载官方 notebook。题面以链接为准，个人实现在 [solutions/](../solutions/README.md)，来源和版权边界在 [UPSTREAM.md](../UPSTREAM.md)。
每道题下面写三件事：官方要交什么、本仓库交了什么、哪一部分留在 [mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra)。数字级的模块对照在 [BRIDGE.md](../BRIDGE.md)。

## 和 mlsys-infra 的区别与联系

两个仓库问的是同一句「算得过来、放得下、传得过来」，机器不一样，所以求解和消费要分开。

区别有四条。

1. **机器。** 这边是 CPU、MCU、NPU：LPDDR 或 SRAM 的字节、int32 累加、定点 requant。那边是 H100 / A100：HBM、Tensor Core、多卡和在线服务。本机拐点是 11.5 FLOP/byte，H100 的屋顶拐点是 295 FLOP/byte，两套数不能拿来互相解释。
2. **量化谁求解、谁执行。** 幅度剪枝、Wanda、GPTQ、AWQ、SmoothQuant、LLM.int8 的求解在 mlsys-infra 的 [编译与压缩](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/README.md)。这边只消费它们产出的布局：ORT 的 QLinearConv、llama.cpp 的 Q4_0 / Q4_K_M，并核对执行语义（逐位、逐字节、困惑度）。
3. **编译器做到哪一层。** 这边不安装 TVM。调度搜索是自己生成 C++、在真机上测量，噪声和最优调度是否可迁移都写进结果。那边讲融合、`torch.compile` 和 GPU kernel。
4. **协议相同才能对表。** GPT-2 的 fp32 困惑度两边用同一协议（wikitext-2、1024 token 窗口），可以直接对。Qwen2.5-7B 的绝对困惑度不能对：那边是 Hugging Face、ctx 1024，这边是 llama-perplexity、ctx 512，只比较相对涨幅。

联系也有四条，约定写在 [BRIDGE.md](../BRIDGE.md) 的第一张表里：结果文件格式、计时分位、`claims.json`、报告和笔记校验器是同一套。公式是同一条，只换带宽和累加器。面试时从任一侧切到另一侧的四句问法也在 BRIDGE 末节。

下面每道官方题都按这个边界处理。标成「不重做」的，是已经在 mlsys-infra 里有实测的求解问题，不是还没写完。

## MIT 6.5940

课程页写的五次 lab：剪枝、量化、神经架构搜索、LLM 压缩、在笔记本上部署 LLM。2025 秋因教师学术休假未开设。

### Lab 1　剪枝

官方要的是幅度剪枝和结构化稀疏之后精度掉多少。本仓库没有稀疏 GEMM，不重做。

mlsys-infra 在 GPT-2 124M 上测过一次性剪枝：50% 稀疏时幅度剪枝困惑度 538.35，Wanda 47.21；2:4 时幅度 735.95，Wanda 116.38。同一份实验还量了 2:4 cuSPARSELt 在大 GEMM 上 1.54–1.58 倍、M ≤ 2048 时更慢。端侧面试如果被问到「稀疏能不能加速小模型」，用的是这组数，不是再剪一次 MobileNet。

### Lab 2　量化

官方用 VGG 做 k-means 和线性量化。这边换成 CIFAR-10 上 20.9 万参数的 MobileNetV2 式网络，因为端侧量化的难点是 depthwise、线性瓶颈和残差，不是 VGG 的 3×3 堆叠。

交了两处实现。[lab2-quantization](../solutions/mit6.5940/lab2-quantization/README.md) 做粒度、五种校准、敏感度、混合精度、QAT 和 ORT 部署；五个种子的配对差写在同一篇 README 里。[int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) 做官方 notebook 停在模拟量化之前的那一段：int8×int8→int32、定点 requant 与 ORT 逐元素比对、VNNI 打包和 AMX。

LLM 的离群通道和 GPTQ 不在这次 lab 里，见 Lab 4。

### Lab 3　神经架构搜索

官方是在延迟或 SRAM 约束下搜一个网络。本仓库不跑搜索。一次搜索给出的是某个空间、某份预算下的一个网，换预算结论就变。模块 01 和模块 04 保留的是对任何网都成立的那笔账：SRAM 是硬约束，激活峰值有下界，patch 用重算换峰值。mlsys-infra 同样不做 NAS。

### Lab 4　LLM 压缩

官方是 GPTQ、AWQ、SmoothQuant 这一类权重要么从数据里求出来。求解在 mlsys-infra，不在这里重做。那边 GPT-2 124M 上 W4 g128 的困惑度涨幅是 RTN +11.1%、AWQ +8.0%、GPTQ +4.1%；Qwen2.5-7B 上 W4 g128 是 RTN +5.9%、AWQ +3.8%、GPTQ +2.9%（Hugging Face，ctx 1024）。

这边用 llama.cpp 消费同类布局。[llama-cpp-7b](../solutions/edge-runtime/llama-cpp-7b/README.md) 里 Q4_K_M 加 imatrix 的相对涨幅是 +1.2%（7.1016 → 7.1846，llama-perplexity，ctx 512）。imatrix 和 AWQ / GPTQ 一样按激活给误差加权，协议不同，只比涨幅。

### Lab 5　在笔记本上部署 LLM

官方走 AWQ 加 TinyChat。这边不复现 TinyChat 的 CUDA kernel。部署问题改成块量化的字节、GEMV 能拿到多少带宽、整网和「层时间之和」差多少，实现是 [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) 和 [llama-cpp-7b](../solutions/edge-runtime/llama-cpp-7b/README.md)。CPU 上 Q8_0 最快，GPU 上 Q4_K_M 最快，这个排序官方 notebook 里没有。

## Machine Learning Compilation

作业在 `mlc-ai/notebooks` 的 `assignment/`。本仓库不安装 `mlc-ai-nightly`，不把 notebook 拷进来。

### Assignment 1　端到端模型执行

官方四段：准备一个 Fashion-MNIST 的小 CNN；用 Relax `BlockBuilder.emit_te` 把 PyTorch 模型迁成 IRModule；把卷积注册成外部库函数；再对这个 IRModule 里的 conv2d 做 schedule。

这四段的机制在本仓库是拆开测的，没有合成一个 TVM IRModule。

| 官方的那一步 | 本仓库里对应的实验 |
| --- | --- |
| 模型从框架迁到可执行的图 | [lab2](../solutions/mit6.5940/lab2-quantization/README.md) 导出 ONNX，ORT 跑同一张图；[tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md) 用 `torch.fx` 读出每个激活的生死 |
| 一层交给外部库 | [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) 用 ORT 的 `QLinearConv` 做逐位参照；[q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) 的打包与 gguf 逐字节一致 |
| 在整张图里变换其中一层 | [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md)：自己生成 C++，在真机上搜调度 |

把同一张网交给几套运行时、包括 TVM Relax 的那一次端到端对照，在 [runtime-compare](../solutions/edge-runtime/runtime-compare/README.md)，见文末。

### Assignment 2–4　TensorIR、自动优化、集成

官方在 TVM 里写 TensorIR、跑 AutoTVM / Ansor、再接回编译器。这边用 [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md) 把闭环做完：30,240 个调度点、四种搜索、岭回归代价模型、复测、256 / 512 位向量。不调用 TVM。和官方实现的差别写在该实验 README 的「和原课程的差异」里：没有 sketch 生成，没有 XGBoost，每个点都是实测。

runtime-compare 量出了不调优的 TVM 有多慢：同一张 micronet，TVM 0.27 不调优要 4107 µs，PyTorch eager 1173 µs，ORT 202 µs。也就是说，作业 2–4 的调优正是 TVM 追上部署运行时所缺的那一步。在无卡容器上用 MetaSchedule 调这张网没有调完：24 个任务，第一个任务的第一批 62 个候选 15 分钟没量完，原因记在该 README 的第 8 条。

## Harvard TinyML 与 MLPerf Tiny

HarvardX / CS249r 的视觉唤醒词和关键词唤醒，定义的是 256 KB 级 SRAM 上的任务。本仓库不下载语音和图片，也不训练 KWS。容量账在 [tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md)：MBv2-0.35@160 的激活峰值 384,000 B，前 4 个 block 切 3×3 patch 后 102,400 B。要做唤醒词本身，4 核 CPU 和公开数据集就够，不为它租服务器。

MLPerf Tiny 要求精度、延迟、峰值 SRAM、Flash 四列一起报。这四列在本仓库里已经有数，但是两个模型、两种设备假设，不能拼成一份 MLPerf 提交。

| 列 | 数字 | 出自 |
| --- | --- | --- |
| 精度 | fp32 88.16%，ORT int8 QDQ 88.09% | lab2 的 micronet，CIFAR-10 |
| 延迟 | 单线程 372.4 µs → 253.9 µs | 同上，Ice Lake 虚拟机上的 ORT 1.31 |
| Flash | ONNX 824 KiB → 319 KiB | 同上，文件大小 |
| 峰值 SRAM | 384,000 B → patch 后 102,400 B | tinyengine 的 MBv2-0.35@160，int8 激活，不是 micronet |

## 对照：同一张网，几套运行时

实验在 [runtime-compare](../solutions/edge-runtime/runtime-compare/README.md)。2026-11-12 在 AutoDL 无卡容器（Xeon 8470Q，cgroup 0.5 核、2 GB）上，折叠后的 micronet 走了 6 条路径，每条 1 个线程，跑完 10,000 张测试图，3 个新进程取中位数，时间是 `process_time`：

| 路径 | top-1 | 和 PyTorch 的 argmax 一致 | CPU 时间 / 图 |
| --- | ---: | ---: | ---: |
| PyTorch 2.8 eager | 88.16% | 1 | 1173 µs |
| ORT 1.31 fp32 | 88.16% | 1 | 202 µs |
| ORT 1.31 int8 QDQ | 88.09% | 0.9905 | 127 µs |
| LiteRT 2.3 | 88.16% | 1 | 166 µs |
| ExecuTorch 1.6 XNNPACK | 88.16% | 1 | 192 µs |
| TVM 0.27（不调优） | 88.16% | 1 | 4107 µs |

四个 fp32 运行时和 PyTorch 的最大 logit 差都是 5.7e-6。ORT int8 的精度和文件大小（326,171 B）与 lab2 一致，CPU 时间是 ORT fp32 的 1/1.60。

在 0.5 核、2 GB 上把这 6 行跑通，要处理 8 个环境问题：三个运行时的 torch 版本互斥，所以分三个 venv；TVM 只要能导入 torch 就会现编 C++ 扩展；apache-tvm-ffi 0.1.15 和 0.1.16 序列化时段错误；页缓存会撑满 cgroup。逐条写在 runtime-compare 的 README 里。

## 官方页面

- 6.5940 课程表与 lab 列表：https://hanlab.mit.edu/courses/2024-fall-65940
- MLC 作业目录：https://github.com/mlc-ai/notebooks/tree/main/assignment
- TinyML 任务列表：https://mlsysbook.ai/courses/tinyml/
