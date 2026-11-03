# 阅读路线

八个模块都已有笔记和实验。按岗位挑路线，每条路线先读笔记、跑实验，再看对应的面试材料；公式编号见 [FORMULAS](../interview/FORMULAS.md)。
官方作业（MIT 6.5940、MLC）在官方页面上自己做，边界见 [assignments](../assignments/README.md)。数据中心一侧的路线在 mlsys-infra 的 [READING-ROADMAP](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/READING-ROADMAP.md)。

## 公共部分（所有岗位，约 1 周）

1. [M01](module-01-edge-roofline.md)：跑 [host-roofline](../solutions/edge-runtime/host-roofline/README.md)，记住本机的拐点和单核带宽（E1–E3）。
2. [M02](module-02-integer-contract.md)：跑 [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) 的测试，能默写 E4–E6。
3. [HARDWARE-NUMBERS](../interview/HARDWARE-NUMBERS.md) 的十个数。

## 部署 / 量化

1. [M03](module-03-calibration-localization.md) 与 [lab2-quantization](../solutions/mit6.5940/lab2-quantization/README.md)：校准、定位、混合精度、QAT（E7–E9）。
2. 作品集故事 [1](../interview/PORTFOLIO.md#1-kl-校准掉-24-个点) 和 [3](../interview/PORTFOLIO.md#3-三方整数逐位对齐)，题库 Q5–Q12。
3. mlsys-infra 的 [M17](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)：同一套问题在 LLM 上（GPTQ、AWQ、SmoothQuant）。
4. 有板子时再加 RKNN 或 TensorRT 一列，用 [ENGINEERING](../interview/ENGINEERING.md) 第 2 节的整数验收标准。

## MCU

1. [M04](module-04-static-memory-planning.md) 与 [tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md)：下界、规划器、原地 depthwise、patch（E10–E14）。
2. [M07](module-07-dma-double-buffering.md) 与 [dma-double-buffer](../solutions/edge-runtime/dma-double-buffer/README.md)：Dory 式分块（E15–E18）。
3. CMSIS-NN 文档里 int8 卷积的 scale 约定，对照 M02 的 E5。
4. 设计题 [D1](../interview/SYSTEM-DESIGN.md#d1-256-kb-sram-的-mcu-上做视觉唤醒词)，作品集故事 4。

## NPU 编译 / mapping

1. [M06](module-06-schedule-search.md) 与 [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md)：调度空间、寄存器块、代价模型（E22–E24）。
2. [M07](module-07-dma-double-buffering.md)：双缓冲与分块搜索；depthwise 从片外供数是 DMA 受限。
3. mlsys-infra 的 [M10](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-10-ml-compilers.md)：同一个搜索在 H100 的解析模型上。
4. 设计题 [D3](../interview/SYSTEM-DESIGN.md#d3-模型上-npu有两个算子不支持)，作品集故事 5。

## CPU 推理引擎（ncnn、MNN、XNNPACK）

1. [M05](module-05-convolution-kernels.md) 与 [conv-kernels](../solutions/edge-runtime/conv-kernels/README.md)：im2col、implicit GEMM、depthwise 布局、Winograd（E19–E21）。
2. M02 的 VNNI 打包与 requant 融合，作品集故事 2。
3. 读 ncnn 或 MNN 的卷积分派：直接卷积、im2col GEMM、depthwise、Winograd 各在什么尺寸下被选中，和 conv-kernels 的结果对照。

## 端侧 LLM

1. [M08](module-08-block-quantized-decode.md) 与 [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md)：块格式、解码上界、整数点积（E25–E27）。
2. mlsys-infra 的 [M15](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-15-inference-kv-speculative.md)（KV、推测解码）和 [M17](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)（GPTQ、AWQ、离群通道）。
3. 设计题 [D2](../interview/SYSTEM-DESIGN.md#d2-手机上跑-1b-级-llm目标-20-tokens)，作品集故事 6。
4. 读 llama.cpp 的 `ggml-cpu` 里 `vec_dot_q4_0_q8_0` 的 AVX2 与 NEON 实现，和本仓库的 `gemv_q4_q8_vnni` 逐条对照。
