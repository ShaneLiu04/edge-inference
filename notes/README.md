# 端侧部署八讲

原创中文笔记，保留英文术语。顺序从「一台设备能做什么」到「怎么把模型塞进去、跑起来」：先 roofline 与整数语义，再量化、内存、内核，最后搜索、搬运和 LLM 解码。
每讲标出它在 MIT 6.5940、MLC、TFLite Micro / CMSIS-NN、PULP Dory、llama.cpp 里的出处，取舍理由见 [CURRICULUM.md](../CURRICULUM.md)。
每讲都有一个可在 4 核 CPU 上跑完的实验，笔记里的数字由 [`claims.json`](../claims.json) 对照原始结果校验。

数据中心一侧的同一套问题在姊妹仓库 [mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/README.md)（十八讲）。两边的对应关系见 [BRIDGE.md](../BRIDGE.md)。

## 课程索引

| # | 部分 | 主题 | 笔记 | 实验 | mlsys-infra 对应 |
| ---: | --- | --- | --- | --- | --- |
| 1 | 设备与语义 | 端侧 roofline：带宽、峰值、ridge、设备表 | [M01](module-01-edge-roofline.md) | [host-roofline](../solutions/edge-runtime/host-roofline/README.md) | M01 |
| 2 | 设备与语义 | 整数部署契约：requant、VNNI、ORT 逐位对齐 | [M02](module-02-integer-contract.md) | [int8-kernels](../solutions/edge-runtime/int8-kernels/README.md) | M02、M17 |
| 3 | 量化 | 校准与误差定位：粒度、KL、敏感度、混合精度、QAT | [M03](module-03-calibration-localization.md) | [lab2-quantization](../solutions/mit6.5940/lab2-quantization/README.md) | M17 |
| 4 | 内存 | 静态内存规划：存活下界、arena、原地 depthwise、patch | [M04](module-04-static-memory-planning.md) | [tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md) | M14 |
| 5 | 内核 | 卷积实现：im2col、implicit GEMM、depthwise NHWC、Winograd | [M05](module-05-convolution-kernels.md) | [conv-kernels](../solutions/edge-runtime/conv-kernels/README.md) | M07 |
| 6 | 编译 | 调度搜索：空间、寄存器分块、代价模型、测量噪声 | [M06](module-06-schedule-search.md) | [tensor-program-tuning](../solutions/mlc/tensor-program-tuning/README.md) | M10 |
| 7 | 搬运 | DMA 与双缓冲：闭式时间、Dory 分块、本机重叠实测 | [M07](module-07-dma-double-buffering.md) | [dma-double-buffer](../solutions/edge-runtime/dma-double-buffer/README.md) | M07、M13 |
| 8 | LLM | 块量化解码：Q4_0 / Q8_0、带宽上界、整数点积、困惑度 | [M08](module-08-block-quantized-decode.md) | [q4-gemv](../solutions/edge-runtime/q4-gemv/README.md) | M15、M17 |

## 每讲结构

与 mlsys-infra 相同的 16 节：摘要、一页速览卡、学习目标、先修知识、核心概念、关键推导、深入推导、白板估算、实现映射、易错点与反例、实验或练习、高频问题、面试追问链、自测题、关联笔记、本讲小结、延伸阅读。
白板估算用本机实测参数和 [设备表](../interview/HARDWARE-NUMBERS.md) 里的端侧芯片参数；元数据里的「实测结论」一行是本讲最该记住的数字。

## 辅助导航

- [术语与符号表](GLOSSARY.md)
- [按岗位和时间预算的阅读路线](READING-ROADMAP.md)
- [公式卡 E1–E27](../interview/FORMULAS.md)
- [硬件参数速查](../interview/HARDWARE-NUMBERS.md)

## 笔记使用法

1. 首读「摘要 → 速览卡 → 学习目标」，确定这一讲解决的是容量、带宽、算力还是精度问题；
2. 二读「核心概念 → 关键推导 → 深入推导」，合上笔记重推公式；
3. 做「白板估算」：用设备表的参数算出数字，再对答案；
4. 打开「实现映射」里的实验，跑测试，改一个参数看数字怎么变；
5. 用「面试追问链 → 自测题」做输出式复习；
6. 面试前只过「速览卡 + 白板估算 + 追问链」。

## 校验

```bash
python notes/check_notes.py            # 结构 / 公式 / 链接（含 mlsys-infra 链接）/ 索引 / 术语
python notes/check_notes.py --verbose  # 每讲行数、公式块数、问题数
python scripts/check_claims.py         # 笔记与 README 里的数字对原始结果
```

笔记只写原创摘要与短引用，不复制官方材料。
