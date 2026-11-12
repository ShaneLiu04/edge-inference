# 进度

每完成一项，链接到对应的笔记、实验或报告，而不只是打个勾。没做完的事也写清楚卡在哪里。

## 建仓（2026-11-03）

- [x] 合并 MIT 6.5940、MLC 与 CMSIS-NN / TFLite Micro / ORT / Dory / llama.cpp 的公开契约，排成 8 个模块，写在 [CURRICULUM.md](CURRICULUM.md)
- [x] 8 讲笔记，每讲 16 节（速览卡、推导、白板估算、追问链、自测题），`python notes/check_notes.py` 通过；[术语表](notes/GLOSSARY.md)、[阅读路线](notes/READING-ROADMAP.md)
- [x] 面试材料：[速查表](interview/HARDWARE-NUMBERS.md)、[公式卡 E1–E27](interview/FORMULAS.md)、[设计题](interview/SYSTEM-DESIGN.md)、[作品集](interview/PORTFOLIO.md)、[手写题](interview/CODING.md)、[题库](interview/QUESTION-BANK.md)、[课程问答](interview/COURSE-QA.md)、[工程手册](interview/ENGINEERING.md)
- [x] 白板估算计算器 [`tools/edge_calc.py`](tools/edge_calc.py) 和共享 harness [`edge_harness/`](edge_harness/)，带测试
- [x] 统一的测试入口 `scripts/run-all-tests.ps1`、仓库检查 `scripts/check-repo.ps1`、数字校验 `scripts/check_claims.py`
- [ ] 镜像官方 starter：公司网络访问 github.com 返回 407，来源记在 [UPSTREAM.md](UPSTREAM.md)

## 实验：4 核 Ice Lake 虚拟机（2026-11-03）

| 模块 | 笔记 | 实验 | 报告 | 做了什么 |
| ---: | --- | --- | --- | --- |
| 01 roofline | [M01](notes/module-01-edge-roofline.md) | [host-roofline](solutions/edge-runtime/host-roofline/README.md) | [PDF](reports/host-roofline/host_roofline_report.pdf) | 1 / 4 线程带宽扫描、fp32 与 VNNI 峰值、设备表 |
| 02 整数契约 | [M02](notes/module-02-integer-contract.md) | [int8-kernels](solutions/edge-runtime/int8-kernels/README.md) | [PDF](reports/int8-kernels/int8_kernels_report.pdf) | requant 与 ORT 逐位对齐、maddubs 饱和、8 种 GEMM、ORT 加速比 |
| 03 校准定位 | [M03](notes/module-03-calibration-localization.md) | [lab2-quantization](solutions/mit6.5940/lab2-quantization/README.md) | [PDF](reports/lab2-quantization/lab2_quantization_report.pdf) | 粒度、5 种校准、敏感度、SQNR 定位、混合精度、QAT、ORT 部署 |
| 04 内存规划 | [M04](notes/module-04-static-memory-planning.md) | [tinyengine-memory](solutions/mit6.5940/tinyengine-memory/README.md) | [PDF](reports/tinyengine-memory/tinyengine_memory_report.pdf) | 下界、4 种规划器、原地 depthwise、patch 推理 |
| 05 卷积 | [M05](notes/module-05-convolution-kernels.md) | [conv-kernels](solutions/edge-runtime/conv-kernels/README.md) | [PDF](reports/conv-kernels/conv_kernels_report.pdf) | 7 种实现计时、Winograd 数值 |
| 06 调度搜索 | [M06](notes/module-06-schedule-search.md) | [tensor-program-tuning](solutions/mlc/tensor-program-tuning/README.md) | [PDF](reports/tensor-program-tuning/tensor_program_tuning_report.pdf) | 4 种策略 × 3 种子、代价模型 Spearman、复测、向量宽度 |
| 07 双缓冲 | [M07](notes/module-07-dma-double-buffering.md) | [dma-double-buffer](solutions/edge-runtime/dma-double-buffer/README.md) | [PDF](reports/dma-double-buffer/dma_double_buffer_report.pdf) | 事件模拟、Dory 式搜索、本机两线程实测 |
| 08 块量化 | [M08](notes/module-08-block-quantized-decode.md) | [q4-gemv](solutions/edge-runtime/q4-gemv/README.md) | [PDF](reports/q4-gemv/q4_gemv_report.pdf) | gguf 逐字节一致、9 种 GEMV、GPT-2 280 窗口困惑度 |

## 实验：服务器复测与 7B（2026-11-04，AutoDL 上的 2 × Xeon 8470Q + RTX 6000D）

跑完已归档、关机。汇总见 [README 的报告表](README.md#报告与服务器实测)。

- [x] 6 个 edge-runtime 实验原样搬到 Xeon 8470Q 重跑（绑 NUMA1 的 16 个物理核，1–16 线程），原始数据在各实验的 `artifacts/raw/xeon8470q/`；34 项指标对照在 [CROSS-HOST.md](interview/CROSS-HOST.md) — [报告](reports/edge-runtime-server/edge_runtime_server_report.pdf)
- [x] int8-kernels 新增 AMX tile GEMM：512³ 上是 VNNI 打包的 3.0 倍，K = 24 时只有 0.6 倍 — [README](solutions/edge-runtime/int8-kernels/README.md)
- [x] q4-gemv 新增 Triton 融合反量化 GEMV（RTX 6000D）：Q4_0 读到实测带宽的 99.5%；只算 Linear 预测 Qwen2.5-7B 解码 fp16 92、Q8 163、Q4 284 tok/s — [README](solutions/edge-runtime/q4-gemv/README.md)
- [x] 第 9 个实验 [llama-cpp-7b](solutions/edge-runtime/llama-cpp-7b/README.md)：llama.cpp 跑完整 Qwen2.5-7B 的 11 种 GGUF，CPU / GPU 解码、prefill、困惑度、KL 散度、batch 吞吐；端到端比 q4-gemv 的预测多一个常数 0.41–0.47 ms/token — [报告](reports/llama-cpp-7b/llama_cpp_7b_report.pdf)
- [x] lab2 在 RTX 6000D 上重训 5 个种子，服务器 CPU 上对每个模型重跑全部量化实验，结论改报配对差与 95% 置信区间 — [README](solutions/mit6.5940/lab2-quantization/README.md#五个种子哪些结论经得起重训)

## 文档（2026-11-12）

- [x] 新结果接进笔记：M01、M02、M03、M05、M06、M07、M08 各加一节服务器、五种子或 7B 的对照
- [x] 每个实验一份 LaTeX 报告（共 10 份），每份带参考文献；`scripts/check_reports.py` 接进 `check-repo.ps1`
- [x] 顶层文档（README、PROGRESS、VERIFICATION、CURRICULUM、BRIDGE、作品集）同步到 9 个实验、105 条 claims

## 还没做的

- [ ] 开发板实测：没有 MCU、NPU、Pi 5、Orin，这些设备的数字是设备表参数与模型推算，标注为上界或模型输出。
- [ ] 性能计数器：本地虚拟机和 AutoDL 容器都没有 perf / ncu 权限，512 位向量变慢的原因、Q4 内核与 K-quant 的端口瓶颈只给出估算。
- [ ] TVM / ExecuTorch / TFLite Micro 的端到端对照；本仓库的搜索器、规划器、块格式是自写的最小实现，与官方实现逐字节或逐位对齐的地方在各 README 写明。
- [ ] llama.cpp 那 0.45 ms/token 的固定成本没有用 nsys 拆到具体算子。

## 复验

```powershell
./scripts/run-all-tests.ps1
python notes/check_notes.py
python scripts/check_claims.py
./scripts/check-repo.ps1
python -m ruff check .
```
