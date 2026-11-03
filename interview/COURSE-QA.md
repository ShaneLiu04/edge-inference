# 八讲一句话答案

每讲三问：定义题、计算题、取舍题。答案控制在一口气说完的长度，展开在对应笔记的「面试追问链」，公式在 [FORMULAS](FORMULAS.md)。
数据中心一侧的十八讲在 mlsys-infra [COURSE-QA](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/interview/COURSE-QA.md)；每讲末尾标出对应的 mlsys-infra 讲次。

## 硬件与整数

**M01 端侧 roofline** · [笔记](../notes/module-01-edge-roofline.md) · 对应 mlsys-infra M01、M02
- 拐点是什么？峰值算力除以带宽（E1），本机 fp32 11.5 FLOP/B、int8 29.5 op/B；强度低于它就受带宽限。
- 4096² 的 fp32 GEMV 在本机单线程最快多少？64 MiB / 13.4 GB/s ≈ 5.0 ms，实测 4.76 ms（14.1 GB/s）。
- 单核带宽低，加 SIMD 还是加线程？加线程。单核受在途请求数限制（E2），4 线程 3.6 倍；SIMD 宽度不增加在途数。

**M02 整数契约** · [笔记](../notes/module-02-integer-contract.md) · 对应 mlsys-infra M17
- int8 卷积的输出怎么算？int32 累加 Σx_q·w_q，减去 z_x·Σw（离线折进 bias），乘定点乘数 M0·2^-(31+shift)，加 z_y、截断到 [0, 255]（E4、E5）。
- shift = 10 的两步舍入有多少输出和单步舍入不同？约 2^-11 ≈ 0.05%（E6），本仓库实测一致率 99.93–99.97%。
- x86 上 int8 点积用 maddubs 还是 VNNI？有 VNNI 就用 VNNI，直接进 int32；maddubs 在 |w| > 64 时 int16 饱和，要把权重限到 7 bit。

## 量化与内存

**M03 校准与误差定位** · [笔记](../notes/module-03-calibration-localization.md) · 对应 mlsys-infra M17
- 逐张量和逐通道的区别？一个 scale 对每个输出通道一个 scale；BN 折叠后通道幅值相差很大，逐张量让小通道接近 0（W4：76.42 对 83.20%）。
- 8 bit 改 6 bit，舍入噪声变多少？步长 4 倍，噪声功率 16 倍（E7、E8 的 s²/12），约少 12 dB。
- 精度掉了，先换校准器还是先定位？先定位：一次只量化一个点或一次只放开一层；KL 在这个网络上掉 24 个点，是零尖峰的问题，换 percentile 只是绕过去。

**M04 静态内存规划** · [笔记](../notes/module-04-static-memory-planning.md) · 对应 mlsys-infra M14
- 激活峰值的下界？每个时刻存活张量之和的最大值（E10）；MBv2-1.0@224 int8 是 1,505,280 B。
- 256 KB SRAM 放得下 MBv2-0.35@160 吗？下界 384,000 B 放不下；前 4 个 block 切 3×3 patch 后 102,400 B，整网 MACs +27%。
- 原地 depthwise 和 patch 推理怎么选？先原地（零重算，省 6.7%）；还不够再 patch，按 E13 估重算，切得越深越贵。

## 算子与编译

**M05 卷积 kernel** · [笔记](../notes/module-05-convolution-kernels.md) · 对应 mlsys-infra M07
- implicit GEMM 和 im2col 的区别？都把卷积当 GEMM 算；im2col 先把 patch 展开到工作区（E19），implicit 在 GEMM 的寻址里现算，工作区为 0。
- ResNet 56×56×64 3×3 fp32 的 im2col 工作区多大？56·56·9·64·4 = 7.2 MB，是输入加输出的 4.5 倍。
- Winograd 用 F(2,3) 还是 F(4,3)？端侧基本 F(2,3)：乘法少 2.25 倍，fp16 下与直接卷积同精度；F(4,3) 误差 7.2 倍，int8 没有余量。

**M06 调度搜索** · [笔记](../notes/module-06-schedule-search.md) · 对应 mlsys-infra M10
- 调度空间由什么组成？循环顺序、三个维度的分块、是否打包、是否展开（E22），256³ 共 30,240 个点。
- t2×64×64 需要几个 ymm 累加器？2·64 / 8 = 16，正好是 AVX2 的寄存器数（E23）。
- 代价模型还是直接测？测量便宜、噪声和差距同量级时直接测并复测；本机代价模型 Spearman 0.47–0.53，不如随机。

## 搬运与解码

**M07 DMA 与双缓冲** · [笔记](../notes/module-07-dma-double-buffering.md) · 对应 mlsys-infra M07、M13
- 双缓冲省的是什么？较短的那一侧：T 从 n·(m + c) 变成 n·max(m, c) + min(m, c)（E15）。
- m = 5、c = 1、4 块 tile，双缓冲多快？24 对 21，只省 12.5%。
- tile 越小越好吗？不是。每块有 DMA 和调用的固定开销（E18），本机两线程时 4 KiB tile 的握手比搬运还长，双缓冲反而慢。

**M08 块量化解码** · [笔记](../notes/module-08-block-quantized-decode.md) · 对应 mlsys-infra M15、M16、M17
- Q4_0 是什么？每 32 个权重一块，16 B 半字节加一个 fp16 scale，共 18 B，4.5 bit/权重（E25）。
- Llama-7B Q4_0 在 Pi 5 上最快多少 tok/s？3.79 GB / 17.1 GB/s → 4.5 tok/s，这是总线峰值给的上界。
- 端侧选 Q4_0 还是 Q8_0？权重从 DRAM 流过时选 Q4（字节少一半）；单核时 Q4 被解包指令限住，只拿到字节比的 65%；GPT-2 上 Q8 困惑度几乎不变（29.953），Q4_0 31.88。
