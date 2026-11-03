# Module 04 — 静态内存规划：存活期下界、arena 碎片、原地 depthwise、按 patch 推理

- 课程出处：MIT 6.5940 Fall 2024《TinyEngine and Parallel Processing》《MCUNet / MCUNetV2》 · TFLite Micro `GreedyMemoryPlanner` · MLC《计算图优化》
- 必读论文：Lin et al., MCUNet: Tiny Deep Learning on IoT Devices, 2020；Lin et al., MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, 2021；Pisarchyk & Lee, Efficient Memory Management for Deep Neural Net Inference, 2020
- 对应实验：[tinyengine-memory](../solutions/mit6.5940/tinyengine-memory/README.md)
- 实测结论：MobileNetV2-1.0@224 int8 存活峰值 1,505,280 B；按大小贪心正好等于下界，first-fit 多 13.3%，每块独占 4.68 倍；原地 depthwise 再省 6.7%（本仓库 micro 网络省 30.8%）；前 4 个 block 切 2×2 patch，峰值降到 566,784 B，整网 MACs 只多 7.7%
- 状态：已验证（数字来自 `artifacts/raw/summary.json`，`claims.json` 校验）

## 摘要

MCU 上激活放在几百 KB 的 SRAM 里，权重在 Flash，所以容量先撞上的是激活的峰值，不是参数量。
推理图是静态的，每个张量的生、死时刻编译期就知道，内存规划等价于一维装箱：给每个张量一个偏移，存活期相交的张量地址不能重叠。
存活字节的最大值是任何规划都打不破的下界；规划器离下界的距离就是碎片。
下界本身还能再压：depthwise 逐通道计算，输出可以覆盖输入（in-place）；前几层的大激活可以按空间 patch 分块跑，用重算 halo 换峰值（MCUNetV2）。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 峰值 = max_t Σ size(存活于 t)；arena ≥ 峰值；arena − 峰值 = 碎片。
2. 规划顺序决定碎片：按出生顺序 first-fit 在 MBv2-1.0@224 上多 13.3%；按大小降序贪心（TFLM 的做法）在本仓库 11 个配置上全部等于下界。
3. MobileNetV2 的峰值在前几层：224 分辨率、int8 时第 2 个 block 的输入加输出就是 1.47 MB，后面层的激活小一个数量级。
4. 原地 depthwise：输出通道 c 只依赖输入通道 c，算完一个通道就能覆盖它，dw 处从「输入 + 输出」降到「max(输入, 输出) + 一个通道」；整网峰值随后移到第二高点。
5. 按 patch 推理：前 k 层逐 tile 执行，每个 tile 的输入要向外扩 receptive field 的 halo，重算量随切分数和层深二次增长。

**必须能默写的公式**

```text
E10  峰值下界：  L = max_t Σ_{i: b_i ≤ t ≤ d_i} s_i
E11  arena 大小：A = max_i (off_i + s_i)，A ≥ L，碎片率 = A / L − 1
E12  halo 回推：  in = (out − 1)·stride + kernel，从 cut 处逐层往输入回推
E13  patch 重算：  stage 开销 = p² · MACs(tile_in) / MACs(全图)
                 整网开销 = (p² · MACs(tile_in) + MACs(其余层)) / MACs(整网)
E14  patch 峰值：  max(stage 输出整块 + 单 tile 在 stage 内的峰值, 其余层的峰值)
```

**一句话区分**

- 下界 vs arena：存活字节的上确界 vs 实际偏移的高水位；差值来自放置顺序。
- 原地 vs 别名：原地是同一块内存先读后写（需要逐通道依赖）；别名是根本不分配新内存（BN、ReLU 融合后、flatten）。
- patch 推理 vs 重计算（rematerialization）：都是用算力换内存；patch 是空间维切分加 halo 重算，训练里的重计算是时间维丢弃再重算。

## 学习目标

- 从一张 fx 图里推出每个张量的生死区间，写出峰值下界；
- 实现 first-fit 与按大小贪心两种放置，并能构造让 first-fit 产生碎片的反例；
- 说清原地 depthwise 成立的条件，以及为什么普通 3×3 卷积不能原地；
- 推导 patch 推理的 halo、峰值与重算比例，并在 nRF52840（256 KB）上给出一个可行配置。

## 先修知识

- M01：SRAM 与 Flash 的容量、带宽量级；
- M02：int8 激活每元素 1 字节，BN 已经折进卷积；
- mlsys-infra M14：训练侧的激活重计算，作为本讲 patch 推理的对照。

## 核心概念

### 1. 张量的生死与别名

`graph.py` 用 `torch.fx` 追踪模型，每个节点一个张量，出生时刻是产生它的节点序号，死亡时刻是最后一个消费者的序号。
BN 折进卷积后不再分配；ReLU/ReLU6 在 int8 kernel 里就是 requant 的 clamp，也不分配；flatten 只改 shape。这三类在图里被合并成别名，否则峰值会被虚增。
残差块的跳连让输入一直活到 add，这是 MobileNetV2 中段峰值的来源。

### 2. 三种放置

- 每块独占（naive）：偏移累加，arena = Σ size。MBv2-1.0@224 是 7,047,312 B，4.68 倍下界。
- first-fit：按出生顺序，每块放在最低的、与所有存活期相交者不冲突的偏移。
- 按大小贪心：先放最大的，再按同样规则放小的。大块先占住低地址，小块填缝。TFLM 的 `GreedyMemoryPlanner` 就是这个顺序。

### 3. 原地 depthwise

depthwise 的输出通道 c 只读输入通道 c。按通道顺序计算、用一个通道大小的临时缓冲存当前结果，算完写回输入的位置，输入和输出共享一块 max(输入, 输出) 大小的内存，外加一个通道的 scratch。
条件有两个：逐通道依赖（groups == C_in == C_out），以及输入在 depthwise 之后不再被读（没有残差从 depthwise 输入引出）。stride 2 也成立，因为一个通道的输出整块写进 scratch 再拷回，不存在行内覆盖顺序的问题。`graph.inplace_depthwise` 按这两个条件合并，与 TinyEngine 的做法一致。

### 4. 按 patch 推理（MCUNetV2）

前 k 层的激活最大但空间维也最大。把 cut 处的输出切成 p×p 个 tile，每个 tile 回推出需要的输入区域（含 halo），逐个 tile 跑完前 k 层，写进 cut 处的完整输出缓冲。
前 k 层的大中间激活永远不会整块存在，代价是相邻 tile 的 halo 被重复计算。

## 关键推导

### 下界为什么只在出生时刻取

存活集合只在某个张量出生时变大，死亡时只会变小，所以 max 一定在某个出生时刻取到。`lower_bound` 只扫 `{b_i}`，复杂度 O(n²)，对几十到几百个张量足够。

### first-fit 的碎片反例

四块 (出生, 死亡, 大小)：A(0,0,3)、B(0,1,3)、C(1,1,2)、D(1,1,2)。时刻 0 存活 A+B = 6，时刻 1 存活 B+C+D = 7，下界 7。

```text
first-fit（按出生顺序）：A@0  B@3  C@0（A 已死）  D：[2,3) 只有 1 字节空隙 → D@6   arena 8
更好的放置：            B@0  A@3  C@3            D@5                          arena 7
```

碎片来自 A 先占了低地址：A 死后留下 3 字节的洞，B 卡在 [3,6) 中间，C、D 各 2 字节放不进同一侧。这个例子在 `tools/tests/test_edge_calc.py::test_first_fit_can_fragment`。

### 按大小贪心为什么在链式网络上几乎总是最优

MobileNet 一类网络是近似链式的，存活期是一串短区间，大块之间很少「错位重叠」。大块先放在 0 偏移，后面小块只要不与同时存活的大块冲突即可，高水位由最大的那一对相邻激活决定，而这正是下界取到的位置。
这是经验规律不是定理：区间图着色（存活区间不带大小）是多项式可解的，带大小的动态存储分配是 NP-hard，贪心只是在这一类图上碰巧贴住下界。

## 深入推导

### halo 回推与重算比例

一层卷积 kernel k、stride s，输出宽 w_out 需要输入宽 (w_out − 1)·s + k。cut 在第 k 层后，从 tile 输出宽 t = ⌈H_out / p⌉ 逐层回推（`patch.input_extent`）。

MobileNetV2-1.0@224，cut=4（前 4 个 block，输出 56×56×24）：
- p=2：t = 28，回推到输入约 127 像素（全图 224），每个 tile 覆盖输入的 (127/224)² ≈ 32%，4 个 tile 共 1.29 倍输入面积；
- stage 的 MACs 开销 1.306 倍（实测，`patch.csv`），因为越深的层 halo 越小，开销低于输入面积比；
- 整网 MACs 开销 1.077 倍：前 4 个 block 只占整网 MACs 的约四分之一，重算被稀释。

```text
整网开销 = 1 + (stage 开销 − 1) · MACs(stage) / MACs(整网)
         = 1 + 0.306 · f      实测 1.077 → f ≈ 0.25
```

这是面试里最容易说错的一点：论文和很多博客报的是 stage 内的开销，换成整网口径要乘 stage 的 MACs 占比。本仓库两列都报。

### patch 峰值由谁决定

E14 的两项：stage 输出缓冲（56×56×24 = 75,264 B）加单 tile 在 stage 内的峰值，以及 cut 之后剩余层的峰值。
MBv2-1.0@224 cut=4：p=2 时 566,784 B，p=3、p=4 都是 564,480 B，不再下降，因为峰值已经转移到 cut 之后的层，再切只增加重算。
要继续降，必须把 cut 往后移：cut=6、p=4 是 326,144 B，整网开销 1.59 倍；cut=8、p=4 是 473,824 B 反而更高，因为 stage 输出缓冲本身变大，而 tile 的 halo 已经接近全图（tile_in 被 clamp 到 224）。

### 原地 depthwise 的节省上界

一个倒残差块 expand(1×1) → dw(3×3) → project(1×1)：不原地时 dw 处同时存活 expand 输出和 dw 输出，各 C_e·H·W；原地后只多一个通道 H·W。
节省 = min(输入, 输出) − 一个通道。对整网峰值的影响取决于峰值移到哪：MBv2-1.0@224 的峰值从 dw 处移到同一个 block 的 expand 处，只降 6.7%（1,505,280 → 1,404,928），因为 expand 处的「输入 + expand 输出」本来就只比 dw 处小一点；本仓库的 micro 网络 expand 前的输入相对小，省 30.8%。
所以「原地省多少」不能只看单层，要看峰值移到的第二高点。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的 SRAM 容量：nRF52840 256 KB，STM32F746 320 KB，STM32H747 1 MB。

```text
MBv2-1.0@224 int8，峰值在第二个 block（16 → 96 通道 expand，dw stride 2）：
  dw 处：expand 输出 112×112×96 = 1,204,224 B + dw 输出 56×56×96 = 301,056 B = 1,505,280 B  ← 实测下界
  原地 dw 后这里变成 1,204,224 + 一个通道 scratch 12,544 B，峰值移到 expand 处：
  expand 处：block 输入 112×112×16 = 200,704 B + expand 输出 1,204,224 B = 1,404,928 B   ← 实测原地下界

MBv2-0.35@160：峰值 384,000 B = 375 KiB > 256 KB
  cut=4, p=3 → 102,400 B（100 KiB），整网 MACs 1.27 倍，放进 nRF52840
  cut=6, p=4 → 93,040 B（91 KiB），整网 MACs 2.11 倍，省的 9 KiB 不值 66% 的额外算力
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 生死区间与别名 | `code/memplan/graph.py` `trace` | BN/ReLU/flatten 不分配；残差输入活到 add |
| 下界与三种放置 | `code/memplan/planners.py` | `validate` 检查任意两块存活相交时地址不重叠 |
| first-fit 反例 | `tests/test_memplan.py` | 3、3、2、2 → 峰值 7，first-fit 8 |
| 原地 depthwise | `graph.inplace_depthwise` | 只在 groups == C_in == C_out 时合并 |
| patch 推理 | `code/memplan/patch.py` `patch_plan` | stage 与整网两种重算口径；p=1 退化为整图 |
| 图 | `artifacts/figures/{memory_profile,arena_layout,patch_tradeoff}.png` | 横轴是整网 MACs 开销 |

## 易错点与反例

- 把参数量当内存：MBv2-1.0 的 3.4 M 参数 int8 是 3.4 MB，放 Flash；SRAM 里是 1.47 MB 的激活峰值。
- 忘记别名：把 BN、ReLU 各算一块输出，MBv2 的峰值会虚增一倍以上。
- 只报 stage 内的重算开销：cut=4、p=4 的 stage 开销 1.65 倍，整网只有 1.17 倍；反过来，cut=8、p=4 的 stage 开销 4.99 倍，整网 2.60 倍，都要说清口径。
- 以为切得越细峰值越低：cut=4 时 p 从 3 到 4 峰值不变（564,480 B），瓶颈已移到 cut 之后。
- 普通 3×3 卷积原地：输出位置 (y, x) 依赖输入 (y±1, x±1) 的全部通道，覆盖后邻居读到的是新值。

## 实验或练习

- 假设：按大小贪心在链式网络上贴住下界，first-fit 有 10% 量级碎片。
- 配置：MBv2-1.0 与 0.35 在 96/128/160/224 四种分辨率，加本仓库 micro 网络 32/64/96，共 11 个配置。
- 结果：贪心 11/11 等于下界；first-fit 在 8 个 MBv2 配置上都多 13.3%（同一个 block 结构在每个分辨率上制造同样比例的洞），在 micro 网络上为 0；naive 是下界的 3.6–4.7 倍。
- 练习：把 cut 设为 3、5，画出整网开销 vs 峰值的 Pareto 前沿，找出 nRF52840 上整网开销最小的可行点。

## 高频问题

- 问：TFLM 的 arena 大小怎么定？
  答：编译期跑一遍 `GreedyMemoryPlanner`（按大小降序放置），得到高水位，再加上算子 scratch 和持久缓冲。实践中先给一个大 arena，读 `arena_used_bytes()` 再收紧。
- 问：为什么 MCU 模型的峰值在前几层？
  答：空间分辨率最高，通道数虽少但 H·W 大；MobileNetV2 的 expand 比 6 让第二个 block 的中间激活最大。后面层 H·W 每次下采样缩小 4 倍。
- 问：patch 推理的代价只有算力吗？
  答：还有 stage 输出缓冲必须整块存在、每个 tile 的调度开销，以及 halo 导致的不规则访存；在有 DMA 的芯片上 tile 太小会让搬运开销盖过重算。

## 面试追问链

1. 问：给一张推理图，最少要多少 SRAM？
   答：先算每个张量的生死区间，取 E10 的存活峰值；这是下界，再加 scratch。别名（BN、ReLU、flatten）不能算新张量。
2. 问：你的规划器离下界多远？怎么证明？
   答：报 E11 的碎片率。本仓库 `validate` 检查不重叠，`lower_bound` 给下界；MBv2 上按大小贪心碎片率 0，first-fit 13.3%。
3. 问：下界本身还能压吗？
   答：两条路：改执行方式（原地 depthwise，省输出块）或改执行顺序/粒度（patch 推理，大激活不整块存在）。还能改模型（MCUNet 的 TinyNAS 直接在峰值约束下搜索）。
4. 问：patch 推理重算多少？
   答：E12 回推 halo，E13 算比例。MBv2-1.0@224 cut=4 p=2：stage 1.31 倍、整网 1.08 倍，峰值 1.47 MB → 554 KiB。
5. 问：为什么 p 再大峰值不降了？
   答：E14 的第二项（cut 后剩余层的峰值）成了瓶颈。要继续降得把 cut 后移，但 stage 输出缓冲变大、halo 接近全图，cut=8 反而比 cut=6 差。
6. 问：这和训练里的激活重计算有什么关系？
   答：同一类「算力换内存」。训练里是时间维：前向丢弃激活、反向重算（mlsys-infra M14 的 √n 检查点）；这里是空间维：tile 之间的 halo 重算。两者都要先找到峰值在哪，再决定切哪里。

## 自测题

1. 三块缓冲 4、4、2，存活 [0,1]、[1,2]、[0,2]，下界是多少？first-fit 呢？
   要点：时刻 1 三块都活，下界 10；任何放置都是 10，无碎片。
2. 一层 3×3 stride 2 卷积，输出 tile 宽 14，需要多少输入宽？再往前一层 3×3 stride 1 呢？
   要点：(14−1)·2+3 = 29；(29−1)·1+3 = 31。
3. stage 开销 2.0 倍、stage 占整网 MACs 30%，整网开销多少？
   要点：1 + 1.0 × 0.3 = 1.3 倍。
4. 一个倒残差块有跳连（stride 1，输入输出通道相同），它的 depthwise 能原地吗？
   要点：能。跳连从 block 输入引出，不是从 depthwise 的输入（expand 输出）引出；depthwise 输入只被 depthwise 读，满足条件。

## 关联笔记

- [M01 端侧 roofline](module-01-edge-roofline.md)：SRAM 容量与带宽决定峰值要压到多少
- [M02 整数契约](module-02-integer-contract.md)：int8 激活每元素 1 字节，ReLU 并进 requant 后成为别名
- [M07 DMA 与双缓冲](module-07-dma-double-buffering.md)：有 L1/L2 分层时，tile 还要同时满足搬运重叠
- [mlsys-infra M14 显存优化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-14-memory-optimization.md)：训练侧激活重计算与本讲 patch 重算的对照
- [mlsys-infra M10 ML 编译器](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-10-ml-compilers.md)：静态内存规划是图编译的一个 pass

## 本讲小结

峰值下界由存活期决定，规划器只能逼近它；在链式网络上按大小贪心已经贴住下界，first-fit 有 10% 级碎片。
要突破下界只能改执行方式：原地 depthwise 省一块输出，patch 推理让前几层的大激活不整块存在。
patch 推理的重算必须报整网口径：MBv2-1.0@224 用 7.7% 的整网 MACs 换 2.66 倍峰值下降，再切则收益消失。

## 延伸阅读

- Lin et al., MCUNet: Tiny Deep Learning on IoT Devices, NeurIPS 2020, https://arxiv.org/abs/2007.10319
- Lin et al., MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021, https://arxiv.org/abs/2110.15352
- Pisarchyk & Lee, Efficient Memory Management for Deep Neural Net Inference, 2020, https://arxiv.org/abs/2001.03288
- TFLite Micro `greedy_memory_planner.cc`, https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/memory_planner
