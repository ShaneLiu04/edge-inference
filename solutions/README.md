# 实验

每个实验目录的结构相同：`README.md`（结论、怎么跑、结果、和原课程的差异、不能推出什么、文件表）、`code/` 或 `cpp/`、`tests/`、`scripts/run_experiments.py`、`artifacts/raw/`（原始结果与 `summary.json`）、`artifacts/figures/`。
数字只来自 `artifacts/raw/`；笔记和 README 引用的关键数字由根目录的 [`claims.json`](../claims.json) 校验。官方 starter 不在这里，来源见 [UPSTREAM.md](../UPSTREAM.md)。

| 模块 | 目录 | 语言 | 测试 | 一句话结果 |
| ---: | --- | --- | ---: | --- |
| 01 | [edge-runtime/host-roofline](edge-runtime/host-roofline/README.md) | C++ | 6 | 单线程 DRAM 13.4 GB/s、fp32 155 GFLOP/s、int8 396 GOP/s |
| 02 | [edge-runtime/int8-kernels](edge-runtime/int8-kernels/README.md) | C++ / Python | 33 | 打包 VNNI 对点积版 2.0–30.8 倍；浮点 requant 与 ORT 100% 一致 |
| 03 | [mit6.5940/lab2-quantization](mit6.5940/lab2-quantization/README.md) | Python | 16 | KL 63.68%，去零 88.06%；混合精度 +9.5 KiB 回到 87.69% |
| 04 | [mit6.5940/tinyengine-memory](mit6.5940/tinyengine-memory/README.md) | Python | 9 | 贪心等于下界；patch 2×2 峰值 −62%，MACs +7.7% |
| 05 | [edge-runtime/conv-kernels](edge-runtime/conv-kernels/README.md) | C++ / Python | 14 | implicit GEMM 无工作区；depthwise NHWC 8–9 倍；Winograd 1.16–2.15 倍 |
| 06 | [mlc/tensor-program-tuning](mlc/tensor-program-tuning/README.md) | Python 生成 C++ | 16 | 搜索 2.4 倍于教科书；代价模型 Spearman 0.47–0.53 |
| 07 | [edge-runtime/dma-double-buffer](edge-runtime/dma-double-buffer/README.md) | Python / C++ | 11 | depthwise + HyperRAM 效率 0.257；本机实测 0.61–1.23 倍 |
| 08 | [edge-runtime/q4-gemv](edge-runtime/q4-gemv/README.md) | C++ / Python | 10 | Q4 × Q8 4.6 倍于 fp32；GPT-2 Q4_0 31.88 |
| 09 | [edge-runtime/llama-cpp-7b](edge-runtime/llama-cpp-7b/README.md) | Python（驱动 llama.cpp） | 5 | Qwen2.5-7B 11 种 GGUF；Q4_K_M 的 KLD 是 Q4_0 的 0.55 倍；AMX 重排 prefill 2.8 倍 |
| — | [edge-runtime/runtime-compare](edge-runtime/runtime-compare/README.md) | Python | 5 | 同一 micronet 走 6 条路径：fp32 四个运行时与 PyTorch 逐图同答；ORT int8 127 µs、LiteRT 166、ExecuTorch 192、ORT fp32 202、PyTorch 1173、不调优的 TVM 4107 |

官方作业按课程题面独立实现：

| 作业 | 目录 | 语言 | 测试 | 一句话结果 |
| --- | --- | --- | ---: | --- |
| 6.5940 Lab 1 | [mit6.5940/lab1-pruning](mit6.5940/lab1-pruning/README.md) | Python | 10 | 细粒度 23.6% 尺寸 92.72%；通道 30% 延迟 0.64 倍、92.52% |
| 6.5940 Lab 2 | [mit6.5940/lab2-vgg](mit6.5940/lab2-vgg/README.md) | Python | 6 | k-means 2 bit QAT 22.73 → 92.18%；整数 int8 VGG 92.94% |
| 6.5940 Lab 3 | [mit6.5940/lab3-nas](mit6.5940/lab3-nas/README.md) | Python | 29 | 闭式效率模型与课程计数器 1,004/1,004 一致；预测器 MAE 0.18 pp；250 KB / 60M 92.556% |
| 6.5940 期末项目 | [mit6.5940/project-mcu-e2e](mit6.5940/project-mcu-e2e/README.md) | Python | 5 | 子网 int8 92.53%、估算 223 ms、patch 后 122 KB；mcunet-vww2 91.37%、234 ms、162 KB |
| 6.5940 Lab 4 | [mit6.5940/lab4-awq](mit6.5940/lab4-awq/README.md) | Python | 4 | OPT-1.3B W3：RTN 107.97，AWQ 16.87，加截断 16.04 |
| 6.5940 Lab 5 | [mit6.5940/lab5-tinychat](mit6.5940/lab5-tinychat/README.md) | C++ patch / Python | 6 | W4A8 kernel 28.2 倍，LLaMA2-7B decode 12.5 倍；`-O3` 51.5 倍 |
| MLC 作业 1 | [mlc/e2e-relax](mlc/e2e-relax/README.md) | Python（TVM 0.27） | 6 | 默认 2375 µs → 手写调度 157 → MetaSchedule 44；准确率 83.88% |

另有 `tools/tests`（计算器与 harness，15 个测试）。全量：

```powershell
./scripts/run-all-tests.ps1            # 18 套，206 个测试（没装 TVM 时跳过 3 个）
$env:EDGE_MARCH = "haswell"; ./scripts/run-all-tests.ps1   # 无 AVX-512 的回退路径
```

计时类步骤（`bench`、`measure`、搜索）要在空闲机器上跑；脚本不会并行运行两个计时任务。
