# 官方作业与对照项目

这里不放 starter，也不转载官方 notebook。题面以链接为准，个人实现在 [solutions/](../solutions/README.md)，来源和版权边界在 [UPSTREAM.md](../UPSTREAM.md)。
每道题下面写三件事：官方要交什么、本仓库交了什么、哪一部分留在 [mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra)。数字级的模块对照在 [BRIDGE.md](../BRIDGE.md)。

## 和 mlsys-infra 的区别与联系

两个仓库问的是同一句「算得过来、放得下、传得过来」，机器不一样，所以求解和消费要分开。

区别有四条。

1. **机器。** 这边是 CPU、MCU、NPU：LPDDR 或 SRAM 的字节、int32 累加、定点 requant。那边是 H100 / A100：HBM、Tensor Core、多卡和在线服务。本机拐点是 11.5 FLOP/byte，H100 的屋顶拐点是 295 FLOP/byte，两套数不能拿来互相解释。
2. **量化谁求解、谁执行。** 幅度剪枝、Wanda、GPTQ、AWQ、SmoothQuant、LLM.int8 的求解在 mlsys-infra 的 [编译与压缩](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/README.md)。这边只消费它们产出的布局：ORT 的 QLinearConv、llama.cpp 的 Q4_0 / Q4_K_M，并核对执行语义（逐位、逐字节、困惑度）。
3. **编译器做到哪一层。** 这边的编译器在 CPU 上：MLC 作业 1 用 TVM 0.27 的 Relax 与 MetaSchedule 跑完，调度搜索的闭环另外自己生成 C++、在真机上测量，噪声和最优调度是否可迁移都写进结果。那边讲融合、`torch.compile` 和 GPU kernel。
4. **协议相同才能对表。** GPT-2 的 fp32 困惑度两边用同一协议（wikitext-2、1024 token 窗口），可以直接对。Qwen2.5-7B 的绝对困惑度不能对：那边是 Hugging Face、ctx 1024，这边是 llama-perplexity、ctx 512，只比较相对涨幅。

联系也有四条，约定写在 [BRIDGE.md](../BRIDGE.md) 的第一张表里：结果文件格式、计时分位、`claims.json`、报告和笔记校验器是同一套。公式是同一条，只换带宽和累加器。面试时从任一侧切到另一侧的四句问法也在 BRIDGE 末节。

下面每道官方题都按这个边界处理：官方题面本身在这里按课程协议做完，同类问题在 mlsys-infra 里更大规模的实测作为对照列出。

## MIT 6.5940

课程页写的五次 lab：剪枝、量化、神经架构搜索、LLM 压缩、在笔记本上部署 LLM。2025 秋因教师学术休假未开设，题面与数据以 Fall 2024 为准。Lab 1、2、4、5 的合订报告在 [reports/mit6.5940-labs](../reports/mit6.5940-labs/mit65940_labs_report.pdf)。

### Lab 1　剪枝

官方题：VGG（CIFAR-10）上做细粒度剪枝（敏感度扫描、按层选稀疏度、尺寸 ≤ 25% 时微调回 ≥ 92.5%）和通道剪枝（按下一层权重的 Frobenius 范数排序、MACs 与实测延迟）。
[lab1-pruning](../solutions/mit6.5940/lab1-pruning/README.md)：细粒度剪到 23.6% 尺寸，5 轮微调后 92.72%（稠密 92.95%）；通道剪 30% 时 MACs 0.504 倍、单线程延迟 0.64 倍、微调后 92.52%。细粒度稀疏在稠密 kernel 上不变快（9.25 → 9.18 ms）。

mlsys-infra 在 GPT-2 124M 上测过一次性剪枝：50% 稀疏时幅度剪枝困惑度 538.35，Wanda 47.21；2:4 时幅度 735.95，Wanda 116.38。同一份实验还量了 2:4 cuSPARSELt 在大 GEMM 上 1.54–1.58 倍、M ≤ 2048 时更慢。

### Lab 2　量化

官方题：VGG 上的 k-means 量化（含 QAT 更新中心）与线性量化（BN 融合、整数卷积与线性层）。
[lab2-vgg](../solutions/mit6.5940/lab2-vgg/README.md)：k-means 8 / 4 / 2 bit 经 QAT 为 92.82 / 92.84 / 92.18%；整数 int8 VGG 92.94%，与 fp32 只差 1 张图，int4 PTQ 89.98%。

端侧量化的难点在 depthwise、线性瓶颈和残差，所以另做了两处。[lab2-quantization](../solutions/mit6.5940/lab2-quantization/README.md) 在 CIFAR-10 上 20.9 万参数的 MobileNetV2 式网络上做粒度、五种校准、敏感度、混合精度、QAT 和 ORT 部署；五个种子的配对差写在同一篇 README 里。[int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) 做官方 notebook 停在模拟量化之前的那一段：int8×int8→int32、定点 requant 与 ORT 逐元素比对、VNNI 打包和 AMX。

### Lab 3　神经架构搜索

官方题：在 MCUNet 的 OFA 超网里，用效率预测器（峰值 SRAM、MACs）和 MLP 精度预测器做随机与进化搜索，满足 250 KB / 60M MACs 下 ≥ 92.5%（bonus 200 KB / 30M 下 ≥ 90%），并判断 256 KB / 15M 与 64 KB 是否可行。
超网代码与 VWW 数据来自课程分发的 `mcunetv2-dev` 与 `data` 包；这一题和以它为起点的期末项目（子网 → int8 → 内存规划与 patch → MLPerf Tiny 四列）在做，结果出来后补在这里。
不依赖具体网络的那笔账已经在模块 01 和模块 04：SRAM 是硬约束，激活峰值有下界，patch 用重算换峰值。

### Lab 4　LLM 压缩

官方题：OPT-1.3B 上 W3 g128 的 RTN、保留 1% 显著通道为 fp16（按激活 / 权重 / 随机）、AWQ 缩放搜索，bonus 截断搜索。
[lab4-awq](../solutions/mit6.5940/lab4-awq/README.md)：RTN W3 困惑度 14.47 → 107.97；按激活保留 1% 是 16.69，按权重 103.41、随机 106.79；AWQ 不保留任何 fp16 通道 16.87，加截断 16.04；W4 时 AWQ 14.78。

mlsys-infra 在更大的模型上做了 GPTQ / AWQ 的对照：GPT-2 124M 上 W4 g128 的困惑度涨幅是 RTN +11.1%、AWQ +8.0%、GPTQ +4.1%；Qwen2.5-7B 上 W4 g128 是 RTN +5.9%、AWQ +3.8%、GPTQ +2.9%（Hugging Face，ctx 1024）。
这边另用 llama.cpp 消费同类布局：[llama-cpp-7b](../solutions/edge-runtime/llama-cpp-7b/README.md) 里 Q4_K_M 加 imatrix 的相对涨幅是 +1.2%（llama-perplexity，ctx 512），只比涨幅。

### Lab 5　在笔记本上部署 LLM

官方题：在 TinyChatEngine 里补完五个 W4A8 GEMM kernel（展开、多线程、SIMD、多线程 + 展开、全部），跑 LLaMA2-7B-chat。
[lab5-tinychat](../solutions/mit6.5940/lab5-tinychat/README.md)：x86 AVX2 与 ARM NEON 两条分支都补了，六个实现全部通过官方检查；`all_techniques` 26.09 GOP/s，是参考实现的 28.2 倍，端到端 decode 12.5 倍（14,427 → 1,150 ms/token）。课程 Makefile 是 `-O0`，换 `-O3` 后 kernel 差距 51.5 倍、端到端 191 ms/token。

块量化的字节、GEMV 的带宽、整网与「层时间之和」的差距在 [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) 和 [llama-cpp-7b](../solutions/edge-runtime/llama-cpp-7b/README.md)：CPU 上 Q8_0 最快，GPU 上 Q4_K_M 最快。

## Machine Learning Compilation

作业在 `mlc-ai/notebooks` 的 `assignment/`。课程用的是 2022 年的 `mlc-ai-nightly`；这边迁移到 PyPI 上的 TVM 0.27，不把 notebook 拷进来。

### Assignment 1　端到端模型执行

官方四段：准备一个 Fashion-MNIST 的小 CNN；用 Relax `BlockBuilder.emit_te` 把 PyTorch 模型迁成 IRModule；把卷积注册成外部库函数；再对这个 IRModule 里的 conv2d 做 schedule。
[e2e-relax](../solutions/mlc/e2e-relax/README.md)（报告 [mlc-e2e-relax](../reports/mlc-e2e-relax/mlc_e2e_relax_report.pdf)）：四段全部在 TVM 0.27 上做完，另加 linear0 的调度与 MetaSchedule。五种写法准确率都是 83.88%，与 PyTorch 最大误差 4.2e-6。
默认编译 2375 µs / batch，其中 linear0 占 2344 µs；手写卷积 + linear0 调度 157 µs；MetaSchedule 44 µs；PyTorch eager 505 µs。
MetaSchedule 的调优数据库会因为融合前后 PrimFunc 不同而静默失效，修法和测试写在该 README 的「踩过的坑」第 1 条。

### Assignment 2–4　TensorIR、自动优化、集成

官方在 TVM 里写 TensorIR、跑 AutoTVM / Ansor、再接回编译器。这边用 [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md) 把闭环自己做一遍：30,240 个调度点、四种搜索、岭回归代价模型、复测、256 / 512 位向量。和官方实现的差别写在该实验 README 的「和原课程的差异」里：没有 sketch 生成，没有 XGBoost，每个点都是实测。TVM 自己的 MetaSchedule 在 e2e-relax 里跑完：8 核上 640 次试验 604 s。

runtime-compare 量出了不调优的 TVM 有多慢：同一张 micronet，TVM 0.27 不调优要 4107 µs，PyTorch eager 1173 µs，ORT 202 µs。那次在 0.5 核的无卡容器上 MetaSchedule 没有调完；e2e-relax 在 8 核上调完后，Fashion-MNIST CNN 从 2375 µs 降到 44 µs，说明差的正是调优这一步。

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
