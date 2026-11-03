# 资料索引

只放链接。访问日期 2026-11-03。取舍见 [CURRICULUM.md](../CURRICULUM.md)，版权边界见 [UPSTREAM.md](../UPSTREAM.md)。

## 课程

| 材料 | 链接 |
| --- | --- |
| MIT 6.5940，2024 秋 | https://hanlab.mit.edu/courses/2024-fall-65940 |
| efficientml.ai | https://efficientml.ai/ |
| MLC 中文 | https://mlc.ai/summer22-zh/ |
| MLC 书 | https://mlc.ai/ |
| MLC 作业 | https://github.com/mlc-ai/notebooks/tree/main/assignment |
| HarvardX TinyML 微专业 | https://www.edx.org/certificates/professional-certificate/harvardx-tiny-machine-learning |
| CS249r / TinyML 材料 | https://mlsysbook.ai/courses/tinyml/ |
| CS249r 书稿 | https://github.com/harvard-edge/cs249r_book |

## 要读契约的项目

| 项目 | 链接 | 读什么 |
| --- | --- | --- |
| CMSIS-NN | https://github.com/ARM-software/CMSIS-NN | int8 卷积的 scale 和 bias |
| TFLite Micro | https://github.com/tensorflow/tflite-micro | arena |
| LiteRT 文档 | https://ai.google.dev/edge/litert | 图的导出和委托 |
| llama.cpp | https://github.com/ggml-org/llama.cpp | `q4_0`、`q8_0` 块 |
| llama2.c | https://github.com/karpathy/llama2.c | 一份可以自己重写的前向 |
| TinyEngine | https://github.com/mit-han-lab/tinyengine | SRAM 内的调度 |
| MCUNet | https://github.com/mit-han-lab/mcunet | 配套的模型 |
| MLPerf Tiny | https://github.com/mlcommons/tiny | 报告格式 |
| Dory | https://github.com/pulp-platform/dory | 分块 |
| Deeploy | https://github.com/pulp-platform/deeploy | DMA 与异构内核 |
| ExecuTorch | https://github.com/pytorch/executorch | 分区 |
| ncnn | https://github.com/Tencent/ncnn | 融合与布局 |
| MNN | https://github.com/alibaba/MNN | 后端 |
| RKNN Model Zoo | https://github.com/airockchip/rknn_model_zoo | 有设备时的转换对照 |
| TensorRT 笔记（Lei Mao） | https://leimao.github.io/blog/ | 校准和 engine 检查的写法 |

## 相邻仓库

| 仓库 | 关系 |
| --- | --- |
| [mlsys-infra](../../mlsys-infra/README.md) | GPU 系统、压缩算法、KV 与推测解码 |
| [physical-ai](../../physical-ai/notes/edge-deployment.md) | 动作块相对控制周期的同步和异步 |
