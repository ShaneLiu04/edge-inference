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

另有 `tools/tests`（计算器与 harness，15 个测试）。全量：

```powershell
./scripts/run-all-tests.ps1            # 9 套，130 个测试
$env:EDGE_MARCH = "haswell"; ./scripts/run-all-tests.ps1   # 无 AVX-512 的回退路径
```

计时类步骤（`bench`、`measure`、搜索）要在空闲机器上跑；脚本不会并行运行两个计时任务。
