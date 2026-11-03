# Module 07 — DMA 与双缓冲：什么时候重叠省时间，什么时候省不了

- 课程出处：PULP Dory / Deeploy 的分块与代码生成 · MIT 6.5940《TinyEngine》 · MLC《硬件加速》
- 必读论文：Burrello et al., DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs, 2021；Scherer et al., Deeploy, 2024；Lin et al., MCUNet, 2020
- 对应实验：[dma-double-buffer](../solutions/edge-runtime/dma-double-buffer/README.md)
- 实测结论：事件模拟与闭式 n·max + min 逐点相等；Dory 式搜索里普通 3×3 卷积在三种目标上都是计算受限，双缓冲只多 0.2%–27%；depthwise 从 HyperRAM 供数是 DMA 受限，效率 0.257，双缓冲 1.24 倍封顶；本机两线程「拷贝 + 计算」实测 0.61–1.23 倍，模型 1.00–1.88 倍，差距来自共享缓存、DRAM 与每块一次的跨线程握手
- 状态：已验证（模型与搜索结果在 `artifacts/raw/`，`claims.json` 校验）

## 摘要

MCU 和 NPU 的片上 SRAM 小，权重和激活放在 L2/L3，计算前要用 DMA 把 tile 搬进 L1。
单缓冲时搬运和计算串行；双缓冲让 DMA 搬下一块的同时计算当前块，总时间从 n·(m + c) 降到 n·max(m, c) + min(m, c)。
省下的永远是较短的那一侧：DMA 已经比计算长时，多一块缓冲造不出带宽。
分块搜索要同时满足 L1 容量（双缓冲让输入、权重缓冲翻倍）和每次 DMA、每次核调用的固定开销，所以最优 tile 既不是最大也不是最小。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 双缓冲 = 两块缓冲 ping-pong：DMA 写 B 时计算读 A，交换。L1 占用翻倍。
2. 总时间 E15：n·max(m, c) + min(m, c)。只有 m ≈ c 时接近 2 倍；m ≫ c 时加速比 → 1 + c/m。
3. 搬入、搬出共用一个 DMA 通道时，DMA 一侧是 m_in + m_out，三级流水退化成两级。
4. tile 太小：每次 DMA 的启动开销（几十到上百周期）和每次核调用开销占比上升；tile 太大：放不进 L1，或者流水的填充-排空项变大。
5. 判断受限侧：每个 tile 的 MACs / (MAC/cycle) 和字节 / (B/cycle) 比大小。3×3 卷积几乎总是计算受限，depthwise 和逐元素算子在片外存储上是搬运受限。

**必须能默写的公式**

```text
E15  单缓冲：T1 = n·(m + c)
     双缓冲：T2 = n·max(m, c) + min(m, c)
     加速比：T1/T2 = (m + c) / (max(m, c) + min(m, c)/n) → (m + c)/max(m, c)，n 大时
E16  k 级流水、每级一个引擎、缓冲足够：T = Σ s_i + (n − 1)·max s_i
     与流水线并行的气泡同形：(p − 1)/(m + p − 1)，m 是 micro-batch 数
E17  tile 的 L1 占用：B·(in + w) + out，B 为缓冲数；输入通道被切分时 out 是 int32 部分和，只留一份
E18  每 tile 时间：m = bytes / (B/cycle) + setup_dma，c = MACs / (MAC/cycle) + setup_call
```

**一句话区分**

- 双缓冲 vs 多缓冲：两块足以覆盖「一搬一算」；三块以上只在 DMA 时间抖动或搬入/搬出分开时才有用。
- DMA 受限 vs 计算受限：前者靠减少字节（int8、融合、更好的复用），后者靠更多 MAC/cycle（SIMD、更大的 tile 降低调用开销）。
- 双缓冲 vs cache 预取：一个是软件显式管理的 scratchpad，一个是硬件猜测；前者时间可预测，这是 MCU/NPU 选它的原因。

## 学习目标

- 从时间线推出 E15，并说清 n 有限时的填充-排空项；
- 写一个离散事件模拟器验证 E15、E16 和共享 DMA 通道的情形；
- 在 L1 约束下为卷积搜 tile，解释为什么搜索结果不是最小 tile；
- 判断给定层在给定目标上是 DMA 受限还是计算受限，并给出双缓冲的加速比上限。

## 先修知识

- M01：带宽、算力与 roofline；
- M04：L1/L2 容量和 arena；
- mlsys-infra M13：流水线并行的气泡，与本讲 E16 同形。

## 核心概念

### 1. 时间线

单缓冲：`[m0][c0][m1][c1]…`。双缓冲：`[m0][m1 | c0][m2 | c1]…[c_{n−1}]`，第一块搬运和最后一块计算无法重叠，这就是 min(m, c) 那一项。

### 2. 搬入、计算、搬出三级

有独立的输入、输出 DMA 通道时是三级流水，时间 E16；只有一个通道时，搬入和搬出在同一个引擎上串行，DMA 侧的时间是 m_in + m_out。
本仓库事件模拟器的结果：m=1、o=0.5、n=64、c≤1 时，三级独立引擎约 65，共享 DMA 是 96 = 64 × 1.5，计算在这里完全被藏住，瓶颈是 DMA 通道。

### 3. Dory 式分块

卷积按输出行 th、输出通道 tk、输入通道 tc 切。每个 tile 要在 L1 里放输入切片（含 halo）、权重块、输出切片；双缓冲时前两者 ×2。
切输入通道会产生 int32 部分和：输出 tile 必须常驻、按 4 字节存，直到所有输入通道累加完。所以 tc < C 的方案 L1 代价更高，搜索通常倾向整通道。

### 4. 本机 CPU 上的「DMA」

`cpp/dma.cpp` 用一个线程当 DMA（memcpy 一个 tile 到小缓冲），另一个线程计算，两个原子标志实现 ping-pong。
和 MCU 的区别：两个线程共享 L2/LLC 和 DRAM 带宽，拷贝本身也占用核心的 load/store 端口，所以实测重叠比模型差；MCU 的 DMA 引擎有自己的总线端口，更接近模型。
实测（64 KiB tile、512 块，改变每 tile 的计算量）：双缓冲比单缓冲 0.66–1.23 倍，模型预测 1.00–1.88 倍；计算为 0 时只剩同步开销，反而慢 34%。固定 c/m ≈ 1.2–2.3、改 tile 大小：4 KiB 时 0.61 倍（每块一次跨线程握手：双缓冲每块 1.18 µs，模型只有 0.34 µs，多出的约 0.8 µs 比搬运本身的 0.3 µs 还长），1 MiB 时 1.08 倍。结论和 MCU 一致：tile 太小时固定开销吃掉重叠收益。

## 关键推导

### E15 的推导

第 i 块计算在 max(第 i 块搬完, 第 i−1 块算完) 开始。m ≥ c 时，第 i 块在 (i+1)·m 搬完，计算立刻开始，最后一块在 n·m + c 算完；m < c 时，第一块在 m 搬完，之后计算连续进行，n·c + m。合起来 n·max + min。

### 加速比上界

```text
T1/T2 = (m + c) / (n·max + min) · n
      = (m + c) / (max + min/n)
m = c：       2n/(n + 1) → 2
m = 4c：      5c/(4c + c/n) → 1.25
```

所以「双缓冲能快一倍」只在 m ≈ c 时成立。本仓库 HyperRAM 供数的 depthwise，每 tile DMA 约为计算的 4 倍，搜索得到的加速比 1.24 倍，与 1.25 的上界吻合。

## 深入推导

### 最优 tile 为什么不是最小

把一层的总时间写成 tile 数 N 的函数（双缓冲、计算受限）：

```text
T(N) ≈ N·(MACs/N / P + s_call) + (MACs/N)/P + 输出写回
     = MACs/P + N·s_call + MACs/(N·P) + …
```

N·s_call 随 N 增长，最后一块的排空 MACs/(N·P) 随 N 减小；L1 容量给 N 一个下界（tile 不能太大）。最优点在 L1 约束边界附近，但不一定贴边：本仓库 PULP 目标上 L1 从 16 KB 到 128 KB，3×3 卷积的 tile 数从 256 降到 8，效率从 0.965 升到 0.994。

### 什么时候是 DMA 受限

每个 tile 的计算/搬运比 = (MACs_tile / P) / (bytes_tile / BW)。对 3×3 卷积 MACs_tile ≈ th·W·tk·tc·9，bytes_tile ≈ (th + 2)·W·tc + tk·tc·9，比值约 9·tk·BW/P：PULP（P=16 MAC/cycle，BW=8 B/cycle）上 tk=16 时约 72，远大于 1。
depthwise 没有 tc 维的复用：MACs_tile ≈ th·W·tk·9，bytes ≈ (th+2)·W·tk + tk·9，比值约 9·BW/P。片上 L2（BW=8）时约 4.5，计算受限；HyperRAM（BW≈1）时约 0.56，DMA 受限。这一行推导就是实验里 depthwise 效率从 0.77 掉到 0.257 的原因。

## 白板估算

```text
GAP8 一档集群：8 核 × 2 MAC/cycle（int8 SIMD）≈ 16 MAC/cycle，L2→L1 DMA ≈ 8 B/cycle，L1 64 KB
depthwise 3×3，32×32×96，int8：
  MACs = 32·32·96·9 = 884,736 → 55,296 cycle（计算）
  字节 = 输入 34·34·96 + 输出 32·32·96 = 209,280 B
  L2 供数：209,280 / 8 = 26,160 cycle → 计算受限
  HyperRAM 供数（≈1 B/cycle）：209,280 cycle → DMA 受限，计算只占 26%
  实验：效率 0.257，双缓冲 1.24 倍
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 事件模拟 | `code/tiling/pipeline.py` `simulate` | 随机参数下与 E15、E16 闭式相等；共享 DMA 时 DMA 侧相加 |
| 分块搜索 | `code/tiling/dory.py` `search` | 所有候选满足 E17；双缓冲不慢于单缓冲；L1 变小时 tile 变小、时间变长 |
| depthwise 与片外存储 | `Conv(depthwise=True)`、`TARGETS["pulp-l3-hyperram"]` | DMA 受限，双缓冲加速比不超过 (m + c)/m |
| 本机重叠实测 | `cpp/dma.cpp` `time_serial`、`time_double` | 两线程、原子标志 ping-pong；每行同时输出模型与实测 |
| 图 | `artifacts/figures/{dma_overlap,dory_efficiency}.png` | 横轴 c/m 对数；三个目标各一列 |

## 易错点与反例

- 「开双缓冲延迟减半」：只有 m ≈ c 时接近；HyperRAM 上的 depthwise 只快 1.24 倍。
- 忘记 L1 翻倍：双缓冲的输入和权重缓冲各两份，原本放得下的 tile 可能放不下，搜索得到的 tile 更小、tile 数更多。
- 只看带宽选 tile：不计 DMA 启动和核调用开销时，搜索会选出最小 tile（本仓库早期版本就是这样）。
- 把输出写回当成免费：本模型里输出写回不与计算重叠，pointwise 从 HyperRAM 供数时效率只有 0.82–0.84，主要损失在这里。
- 在 CPU 上测双缓冲就下结论：两个线程共享缓存和带宽，测出的重叠反映的是 CPU 的内存子系统，不是 DMA 引擎。

## 实验或练习

- 假设：普通卷积计算受限，双缓冲收益小；depthwise 在片外存储上 DMA 受限，收益被 (m + c)/m 封顶。
- 配置：四种层（3×3 两种、1×1、depthwise）× 三种目标（PULP + L2、PULP + HyperRAM、Cortex-M7 + DTCM）× L1 16/32/64/128 KB × 单/双缓冲。
- 结果：3×3 卷积在 PULP + L2 上双缓冲只多 0.6%–2.2%，在 HyperRAM 上 5.6%–27%；depthwise 在 HyperRAM 上 DMA 受限，效率 0.257，双缓冲 1.24–1.26 倍。
- 练习：给 `simulate` 加「DMA 时间随机抖动 ±30%」，比较 2、3、4 块缓冲的期望完成时间。

## 高频问题

- 问：双缓冲的代价是什么？
  答：L1 占用翻倍（输入、权重各两份），tile 变小、调用次数变多；还有同步（DMA 完成中断或轮询）的开销。
- 问：为什么 NPU 编译器要做分块搜索而不是固定 tile？
  答：每层的形状、强度不同，计算受限层要大 tile 降低调用开销，搬运受限层要让 DMA 和计算同时进行，L1 约束随层变化；固定 tile 在两头都会损失。
- 问：怎么判断一层是 DMA 受限？
  答：算每个 tile 的 MACs/P 和 bytes/BW，前者小就是 DMA 受限；或者在板上把计算换成空循环，看时间变不变。

## 面试追问链

1. 问：双缓冲能省多少时间？
   答：E15，省 min(m, c) × (n − 1)。加速比 (m + c)/(max + min/n)，m = c 时接近 2，m = 4c 时只有 1.25。
2. 问：搬入和搬出只有一个 DMA 通道呢？
   答：DMA 侧时间变成 m_in + m_out。模拟里 m=1、o=0.5 时从 65 变成 96，计算完全被藏住，瓶颈是通道。
3. 问：tile 怎么选？
   答：枚举 th、tk、tc，过滤 E17 的 L1 约束，用 E18 算每 tile 时间，再用 E15 算总时间；开销项让最优点不在最小 tile。
4. 问：depthwise 为什么特别难？
   答：没有输入通道维的复用，计算/搬运比只有 9·BW/P。片上 L2 供数时还能计算受限，换成片外 HyperRAM 就 DMA 受限，效率掉到 0.257。
5. 问：DMA 受限时还能做什么？
   答：减少字节（int8 甚至 int4 激活；把 depthwise 和前后的 pointwise 融合，让中间激活不出 L1），或者把这一层的输入提前放进 L2（layer fusion / 跨层分块），而不是加缓冲。
6. 问：这和数据中心的什么问题是同一个？
   答：流水线并行的气泡（mlsys-infra M13）：E16 的 Σ s_i + (n − 1)·max s_i 与 GPipe 的填充-排空同形；也和 GPU 上 cp.async 多级流水（M07 的 GEMM）是同一个思想。

## 自测题

1. n = 16，m = 3，c = 1，单缓冲与双缓冲各多少？加速比？
   要点：64；16·3 + 1 = 49；1.31。
2. 一个 tile 输入 12 KB、权重 4 KB、输出 8 KB（int8，不切输入通道），双缓冲要多少 L1？
   要点：2·(12 + 4) + 2·8 = 48 KB（输出也双缓冲，以便搬出与下一块计算重叠）。
3. 为什么输入通道被切分时，输出只留一份但按 int32 存？
   要点：部分和要跨多个 tile 累加，不能搬出；int8 输出要等累加完成后 requant。
4. Cortex-M7 上 3×3 卷积为什么效率接近 1？
   要点：P = 2 MAC/cycle 很低，计算远长于搬运，DMA 和开销都被藏住。

## 关联笔记

- [M01 端侧 roofline](module-01-edge-roofline.md)：每个 tile 的计算/搬运比就是 tile 级的 roofline
- [M04 静态内存规划](module-04-static-memory-planning.md)：L1 容量约束与 arena 规划是同一份预算
- [M06 调度搜索](module-06-schedule-search.md)：分块搜索是调度搜索在显式 scratchpad 上的特例
- [mlsys-infra M13 张量与流水线并行](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-13-tensor-pipeline-parallel.md)：气泡与 E16 同形
- [mlsys-infra M09 加速器](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-09-accelerators.md)：systolic array 的数据搬运与片上缓冲

## 本讲小结

双缓冲省下的是较短的一侧，加速比由 (m + c)/max(m, c) 封顶。
分块搜索在 L1 约束和固定开销之间取舍；普通卷积几乎总是计算受限，双缓冲收益很小，depthwise 在片外存储上才真正需要它，而那时它也救不了带宽。
判断受限侧只需要一行：9·tk·BW/P（卷积）或 9·BW/P（depthwise）和 1 比。

## 延伸阅读

- Burrello et al., DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs, IEEE TC 2021, https://arxiv.org/abs/2008.07127
- Scherer et al., Deeploy: Enabling Energy-Efficient Deployment of Small Language Models on Heterogeneous Microcontrollers, 2024, https://arxiv.org/abs/2408.04413
- PULP Platform, Dory repository, https://github.com/pulp-platform/dory
