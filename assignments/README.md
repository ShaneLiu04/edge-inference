# 官方作业与对照项目

这里不放 starter。题目以链接为准，个人实现在 [solutions/](../solutions/README.md)。来源说明见 [UPSTREAM.md](../UPSTREAM.md)。

| 来源 | 题目 | 本仓库怎么用 |
| --- | --- | --- |
| MIT 6.5940 Lab 1 | 剪枝 | 不重做。幅度剪枝和 Wanda 的测量在 mlsys-infra |
| MIT 6.5940 Lab 2 | 量化 | [lab2-quantization](../solutions/mit6.5940/lab2-quantization/README.md)：换成 MobileNetV2 式网络，做粒度、5 种校准、敏感度、混合精度、QAT 与 ORT 部署；整数执行在 [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) |
| MIT 6.5940 Lab 3 | 神经架构搜索 | 不做独立实验。模块 01 只保留「SRAM 是硬约束」这一个用法 |
| MIT 6.5940 Lab 4 | LLM 压缩 | 不做。GPTQ / AWQ / SmoothQuant 在 mlsys-infra |
| MIT 6.5940 Lab 5 | 笔记本上部署 LLM | 不做 AWQ / TinyChat 本身。模块 08 改成块量化 GEMV 与 GPT-2 困惑度：[q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) |
| MLC assignment 1 | 端到端模型执行 | 模块 06 的前置。重写，不粘贴 `assignment1_zh.ipynb` |
| MLC assignment 2–4 | TensorIR、自动优化、集成 | 不依赖 TVM 重写自动优化闭环：[tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md) |
| Harvard TinyML | 视觉唤醒词、关键词唤醒 | 模块 04 的容量故事：[tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md) 算 MBv2-0.35 能否放进 256 KB。数据集脚本下载，不提交音频和图片 |
| MLPerf Tiny | 四项基准的报告格式 | 精度、延迟、峰值 SRAM、Flash 四列一起报 |

官方页面：

- 6.5940 课程表与 lab 列表：https://hanlab.mit.edu/courses/2024-fall-65940
- MLC 作业目录：https://github.com/mlc-ai/notebooks/tree/main/assignment
- TinyML 任务列表：https://mlsysbook.ai/courses/tinyml/
