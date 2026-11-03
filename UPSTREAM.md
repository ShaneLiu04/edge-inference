# 上游与版权

访问和核对日期：2026-11-03。只记录入口。官方 starter、课件 PDF、板级 SDK 和第三方源码都不镜像进本仓库。

个人实现在 `solutions/` 和 `tools/`。引用数字时引用本仓库跑出来的 `summary.json` 或单元测试里的闭式结果。

## 课程

| 材料 | 入口 | 使用方式 |
| --- | --- | --- |
| MIT 6.5940，2024 秋 | https://hanlab.mit.edu/courses/2024-fall-65940 | 看 Lab 2 的问题定义和 MCUNet / TinyEngine 讲次。不提交、不转载他们的 notebook |
| efficientml.ai | https://efficientml.ai/ | 课程主页 |
| MLC 中文夏校 | https://mlc.ai/summer22-zh/ | 章节顺序 |
| MLC 书 | https://mlc.ai/ | TensorIR、内存、自动调度 |
| MLC 作业 | https://github.com/mlc-ai/notebooks/tree/main/assignment | 自己做。做完的代码若要放进 `solutions/`，应是重写而不是粘贴 notebook |
| HarvardX TinyML | https://www.edx.org/certificates/professional-certificate/harvardx-tiny-machine-learning | 三门：基础、应用、部署 |
| CS249r 材料汇总 | https://mlsysbook.ai/courses/tinyml/ | 任务定义（hello world、手势、视觉唤醒词、唤醒词） |
| CS249r 书稿仓库 | https://github.com/harvard-edge/cs249r_book | 系统教材，边缘只是其中一卷的一部分 |

6.5940 官方实验说明写的五次 lab：Lab 1 剪枝，Lab 2 量化，Lab 3 神经架构搜索，Lab 4 LLM 压缩，Lab 5 在笔记本上部署 LLM。课程页说明 2025 秋因教师学术休假未开设。

## 运行时与项目

读契约和数据布局，用来对照自己的小实现。不把上游源码复制进 git 历史。

| 项目 | 入口 | 对照哪一条契约 |
| --- | --- | --- |
| CMSIS-NN | https://github.com/ARM-software/CMSIS-NN | 对称 int8 权重、per-output-channel scale、int32 偏置 |
| TFLite Micro | https://github.com/tensorflow/tflite-micro | 一张静态 arena |
| LiteRT | https://ai.google.dev/edge/litert | TFLite 的后续名称。文档入口，不替代上面的运行时 |
| ggml / llama.cpp | https://github.com/ggml-org/llama.cpp | Q8_0：32 个 int8 + fp16 scale。Q4_0：32 个 4-bit + fp16 scale |
| llama2.c | https://github.com/karpathy/llama2.c | 一份短到可以自己重写的 Transformer 前向 |
| TinyEngine | https://github.com/mit-han-lab/tinyengine | MCU 上把工作区压进 SRAM 的调度 |
| MCUNet | https://github.com/mit-han-lab/mcunet | 和 TinyEngine 配套的网络搜索 |
| MLPerf Tiny | https://github.com/mlcommons/tiny | 精度、延迟、SRAM、Flash 四栏怎么同时报 |
| Dory | https://github.com/pulp-platform/dory | MCU 自动分块 |
| Deeploy | https://github.com/pulp-platform/deeploy | 异构 MCU 上的 DMA 和内核选择 |
| ExecuTorch | https://github.com/pytorch/executorch | PyTorch 侧的分区和 delegate |
| ncnn | https://github.com/Tencent/ncnn | CPU 视觉模型的融合和 NHWC |
| MNN | https://github.com/alibaba/MNN | 图优化和后端抽象 |
| RKNN Model Zoo | https://github.com/airockchip/rknn_model_zoo | 有瑞芯微板子时的转换对照。工具链不入库 |
| Lei Mao，TensorRT 笔记 | https://leimao.github.io/blog/ | INT8 校准、plugin、engine 检查的写法。NVIDIA 文档仍是行为的依据 |

地平线、昇腾、寒武纪、高通 QNN 的闭源工具链不列入上游镜像。有合法许可和设备时，在本地把同一张误差表的「设备」列填上，数字和脚本分开存放。

## 不入库的内容

- 课程作业的 starter、参考答案、评分脚本
- 任何 SDK 的安装包、license 文件、芯片手册的全文
- 预训练大模型权重
- 含令牌、代理密码或账号的 `tmp_*.py`
