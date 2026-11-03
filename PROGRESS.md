# 进度

2026-11-03：八个模块的笔记、实验、公式卡、claims 全部完成，全部只用 CPU（4 物理核 Ice Lake 虚拟机，AVX-512 + VNNI）。数据中心 GPU 与 GPTQ / AWQ 留在 [mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra)。

| 模块 | 笔记 | 实验 | 状态 |
| ---: | --- | --- | --- |
| 01 roofline | [M01](notes/module-01-edge-roofline.md) | [host-roofline](solutions/edge-runtime/host-roofline/README.md) | 完成：1 / 4 线程带宽扫描、fp32 与 VNNI 峰值、设备表 |
| 02 整数契约 | [M02](notes/module-02-integer-contract.md) | [int8-kernels](solutions/edge-runtime/int8-kernels/README.md) | 完成：requant 与 ORT 逐位对齐、maddubs 饱和、8 种 GEMM、ORT 加速比 |
| 03 校准定位 | [M03](notes/module-03-calibration-localization.md) | [lab2-quantization](solutions/mit6.5940/lab2-quantization/README.md) | 完成：粒度、5 种校准、敏感度、SQNR 定位、混合精度、QAT、ORT 部署 |
| 04 内存规划 | [M04](notes/module-04-static-memory-planning.md) | [tinyengine-memory](solutions/mit6.5940/tinyengine-memory/README.md) | 完成：下界、4 种规划器、原地 depthwise、patch 推理 |
| 05 卷积 | [M05](notes/module-05-convolution-kernels.md) | [conv-kernels](solutions/edge-runtime/conv-kernels/README.md) | 完成：7 种实现计时、Winograd 数值 |
| 06 调度搜索 | [M06](notes/module-06-schedule-search.md) | [tensor-program-tuning](solutions/mlc/tensor-program-tuning/README.md) | 完成：4 种策略 × 3 种子、代价模型 Spearman、复测、向量宽度 |
| 07 双缓冲 | [M07](notes/module-07-dma-double-buffering.md) | [dma-double-buffer](solutions/edge-runtime/dma-double-buffer/README.md) | 完成：事件模拟、Dory 式搜索、本机两线程实测 |
| 08 块量化 | [M08](notes/module-08-block-quantized-decode.md) | [q4-gemv](solutions/edge-runtime/q4-gemv/README.md) | 完成：gguf 逐字节一致、9 种 GEMV、GPT-2 280 窗口困惑度 |

## 没有做的

- 没有开发板：MCU、NPU、Pi 5、Orin 的数字是设备表参数与模型推算，标注为上界或模型输出。
- 没有性能计数器（虚拟机）：512 位向量变慢的剩余原因、Q4 内核的端口瓶颈只给出估算。
- 没有 TVM / ExecuTorch / TFLite Micro 的端到端对照；本仓库的搜索器、规划器、块格式是自写的最小实现，与官方实现逐字节或逐位对齐的地方在各 README 里写明。
- GPT-2 之外没有测更大的 LLM 的困惑度；7B 的解码速度是由单层 GEMV 推算的。

## 复验

```powershell
./scripts/run-all-tests.ps1
python notes/check_notes.py
python scripts/check_claims.py
./scripts/check-repo.ps1
```
