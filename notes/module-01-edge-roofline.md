# Module 01 — 端侧 roofline：先量本机屋顶，再把 kernel 放上去

- 课程出处：MIT 6.5940 Fall 2024《Efficient Inference》与《TinyEngine》 · CMU 15-418 / Stanford CS149《Parallel Computing》 · Williams et al.《Roofline》 · Hennessy & Patterson 第 4–5 章
- 必读论文：Williams, Waterman & Patterson, Roofline: An Insightful Visual Performance Model for Multicore Architectures, CACM 2009；McCalpin, STREAM benchmark, 1995；Ofenbeck et al., Applying the Roofline Model, ISPASS 2014
- 对应实验：[host-roofline](../solutions/edge-runtime/host-roofline/README.md)
- 实测结论：本机（4 核 Ice Lake 服务器虚拟机，AVX-512 + VNNI）1 线程 fp32 FMA 峰值 154.9 GFLOP/s、int8 VNNI 396.1 GOP/s（只有 fp32 的 2.56 倍，不是 4 倍），DRAM 只读 13.4 GB/s，fp32 拐点 11.5 FLOP/B；4 线程 596.6 GFLOP/s、1493 GOP/s、DRAM 48.4 GB/s，单核只能用到 4 核 DRAM 带宽的 28%；L1 到 DRAM 的带宽差 12 倍；本仓库 GEMM 类卷积 kernel 只到屋顶的 16–26%（Winograd 等效 48%），带宽受限的 depthwise NHWC 到 36–87%
- 状态：已验证（`artifacts/raw/roofline.jsonl`、`summary.json`、`lab_points.csv`，`claims.json` 校验）

## 摘要

roofline 把一个 kernel 的性能上界写成 min(峰值算力, 算术强度 × 带宽)。两条屋顶的交点是拐点：强度低于它的 kernel 受带宽限制，高于它的受算力限制。
端侧部署先要回答的不是「这个 kernel 写得好不好」，而是「它在哪条屋顶下面」：depthwise、GEMV、逐元素算子在任何端侧芯片上都在拐点左边，优化方向是少搬字节；3×3 卷积和大 GEMM 在右边，优化方向是喂满 SIMD。
屋顶必须实测：数据手册的峰值带宽是总线算术，单核持续带宽可能只有它的一小半；int8 峰值也不能按位宽从 fp32 换算。本讲在本机量出 1/4 线程的带宽阶梯与两个峰值，再把 conv-kernels、int8-kernels 的实测点画上去。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 屋顶 E1：P = min(P_peak, I·β)，拐点 I* = P_peak/β。本机 1 线程 fp32 拐点 154.9/13.4 = 11.5 FLOP/B，int8 29.5 op/B。
2. 带宽是一张阶梯，不是一个数：1 线程 L1 166 GB/s、L2 约 120、L3 27–40、DRAM 13.4；同一个 kernel 换个工作集大小就换了一条屋顶。
3. 单核喂不满 DRAM：1 线程 13.4 GB/s，4 线程 48.4 GB/s（3.6 倍）。单核持续带宽受在途请求数限制（Little 定律，E2），不是总线宽度。
4. int8 峰值要量：VNNI 每条指令 64 个乘加（fp32 FMA 16 个），按位宽应是 4 倍，实测只有 2.56 倍。
5. GEMV/decode 的强度 ≈ 2/每参数字节数（E3）：fp32 0.5、int8 2、Q4_0 3.6 FLOP/B，全在拐点左边；decode 的速度就是带宽除以权重字节。

**必须能默写的公式**

```text
E1  roofline：   P_att(I) = min(P_peak, I·β)，拐点 I* = P_peak / β
                 I = 运算数 / 从该层存储搬运的字节数（β 换成哪一层的带宽，I 就按哪一层的流量算）
E2  Little 定律： β_sustained = 在途字节 / 访存延迟 = (在途 cache line 数 × 64 B) / latency
E3  强度：       GEMM  I = 2MNK / (b_a·MK + b_w·KN + b_c·MN)
                 GEMV  I = 2NK / (b_w·NK + b_a·(N + K)) → 2 / b_w（N、K 大时）
                 conv  I = 2·MACs / (激活字节 + 权重字节)，每个张量只搬一次的最好情况
```

**一句话区分**

- 峰值 vs 持续：峰值是「每周期能发多少条、每条多宽」的乘积；持续是带依赖、带访存、带调度开销后实际跑出来的。屋顶用实测的持续值，越界的点说明强度算错了。
- 带宽受限 vs 延迟受限：前者加并行度不再变快（总线满了）；后者加在途请求（多线程、预取、展开）还能变快。本机单核 DRAM 读属于后者。
- 拐点左边 vs 右边：左边优化字节（融合、量化、布局、复用），右边优化指令（向量化、寄存器分块、打包）。

## 学习目标

- 给一个算子写出 E3 的强度，判断它在本机与目标芯片上受哪条屋顶限制；
- 设计并解释一组测带宽阶梯与峰值的微基准，知道每个测量会被什么干扰；
- 用 Little 定律解释单核为什么喂不满 DRAM，以及多线程为什么能；
- 用实测峰值反推核心频率与发射端口数，判断一个「峰值」声明是否可信；
- 把一个 kernel 的实测点画到 roofline 上，算效率，并说清下一步优化往哪个方向。

## 先修知识

- 缓存层次、cache line、SIMD 宽度与 FMA 的基本概念；
- mlsys-infra M01：roofline 与拐点在 GPU（H100 拐点 295 FLOP/B）上的推导，本讲是它在端侧 CPU 上的实测版本；
- mlsys-infra M02/M03：多核、SIMD、线程池与调度，本讲的 4 线程测量依赖线程池的派发开销足够小。

## 核心概念

### 1. 两条屋顶与拐点

性能 P（op/s）不可能超过算力峰值 P_peak，也不可能超过「每字节能做多少运算」乘「每秒能搬多少字节」。I 低于 I* 时字节决定时间，高于 I* 时指令决定时间。
屋顶不止两条：带宽有 L1/L2/L3/DRAM 好几层，算力有 fp32 FMA、int8 VNNI、标量几条。哪条屋顶适用取决于数据在哪一层、指令用的是哪一类。

### 2. 带宽阶梯

工作集小于 L1 时每次访问都命中 L1，读带宽受加载端口限制；超过某一级容量后掉到下一级。阶梯的位置告诉你分块该切多大：本机 L1 48 KiB、L2 1.25 MiB/核，1 线程读在 32 KiB 处还是 166 GB/s，64 KiB 掉到 125，2 MiB 掉到 40。

### 3. 峰值的构成

fp32 峰值 = 核数 × 频率 × FMA 端口数 × SIMD lane 数 × 2。AVX-512 一条 FMA 16 lane；Ice Lake 服务器两个 512 位 FMA 端口，每周期 64 FLOP。int8 的 `vpdpbusd` 一条做 64 个 u8×s8 乘加（128 op），但吞吐不等于 FMA，要单独量。

### 4. 实测点与效率

kernel 的实测点 (I, P)：I 按「每个张量只搬一次」的最好情况算，效率 = P / P_att(I)。效率低于屋顶说明还有空间，但「离哪条屋顶近」决定往哪个方向优化。

## 关键推导

### 本机屋顶

```text
1 线程：P_fp32 = 154.9 GFLOP/s   P_int8 = 396.1 GOP/s   β_DRAM = 13.4 GB/s   β_L1 = 166 GB/s
        拐点 fp32 = 154.9 / 13.4 = 11.5 FLOP/B     int8 = 396.1 / 13.4 = 29.5 op/B
        从 L1 供数时 fp32 拐点 = 154.9 / 166 = 0.93 FLOP/B
4 线程：P_fp32 = 596.6   P_int8 = 1492.7   β_DRAM = 48.4   β_L1 = 596（每核 149）
        拐点 fp32 = 12.3 FLOP/B     int8 = 30.8 op/B
```

4 线程时算力与带宽几乎同比放大（3.85 倍与 3.6 倍），拐点基本不变。这是因为 4 个核还没有把 DRAM 总线占满；核再多，带宽会先饱和，拐点会右移。

### 从峰值反推频率

```text
1 线程 154.9 GFLOP/s ÷ (2 端口 × 16 lane × 2) = 2.42 GHz
4 线程 596.6 ÷ 4 = 149.1 GFLOP/s/核 → 2.33 GHz（全核 AVX-512 降频约 4%）
若只有 1 个 FMA 端口 → 4.84 GHz，不可能 → 本机确是双 FMA 端口
```

用同样的方法检查一个「峰值」声明：它要求的频率或端口数不合理，声明就有问题。

### int8 为什么只有 2.56 倍

```text
VNNI 一条 = 64 乘加 = 128 op；FMA 一条 = 16 乘加 = 32 FLOP → 位宽推出 4 倍
实测 396.1 / 154.9 = 2.56 倍
按 2.42 GHz 计：396.1e9 / (128 × 2.42e9) = 1.28 条 vpdpbusd / 周期（FMA 是 2 条 / 周期）
```

int8 的指令吞吐只有 FMA 的 64%（可能来自端口分配或频率档位；只看墙钟时间无法区分，要靠性能计数器或 uops.info 的表）。工程结论是 int8 的算力屋顶要单独量，按位宽乘 4 会把拐点高估 1.6 倍。

## 深入推导

### 单核为什么喂不满 DRAM：Little 定律

带宽 = 在途字节 / 延迟。一个核能同时挂起的 L1 miss 数有限（Ice Lake 每核 L1 有十几个 line fill buffer，L2 硬件预取再多挂一些）。设 DRAM 延迟约 100 ns（典型服务器值，本机未测）：

```text
单核：13.4 GB/s × 100 ns = 1340 B 在途 ≈ 21 条 64 B cache line
4 核：48.4 GB/s × 100 ns ≈ 76 条在途，平均每核 19 条 → 每核在途数没变，总带宽随核数线性涨
```

单核的持续带宽由在途请求数决定，不由总线宽度决定；所以加线程有效，加 SIMD 宽度无效。端侧 SoC 同理：Pi 5 的 LPDDR4X 总线峰值 17.1 GB/s（数据手册算术），单个 A76 核实际能用到多少要量，不能直接拿 17.1 去除权重字节。

### triad 比只读多搬了字节

triad `a = b + s·c` 按 3 个流计字节（读 b、c，写 a），1 线程大工作集 13.8 GB/s。普通存储会先把 a 的 cache line 读进来再写（write-allocate），总线上实际是 4 个流：

```text
按 4 流计：13.8 × 4/3 = 18.4 GB/s > 只读核的 13.4 GB/s
4 线程：49.1 × 4/3 = 65.5 GB/s > 只读 48.4 GB/s
```

只读核没有占满总线：两个读流加一个写流同时在途，每核挂起的 miss 更多。M08 的 fp32 GEMV（4096×4096，64 MiB 权重）同时读多行，4 线程 58.1 GB/s，高于同样 64 MiB 工作集下的只读核 52.7 GB/s；1 线程 14.1 GB/s，与只读核的 14.4 持平。这再次说明单核 13.4 GB/s 是并发上限，不是总线上限；也说明「带宽」要说清是哪种访问模式。

### 本机测量的设计

- 每个线程在自己那一段上重复扫描 reps 次（总量约 64 MiB），而不是每次扫描都派发一次：一次派发要几微秒，比扫一遍 L1 大小的缓冲还长。
- 每次扫描后 `clobber()`：否则编译器可以把对不变内存的重复求和提到循环外。
- 线程池用自旋派发（`edge_harness` 的 `ThreadPool`），空闲时退到 sleep，不在单线程阶段抢核。早期版本每次调用新建线程，4 线程 L1 读只测到约 50 GB/s，现在是 596。
- 峰值核用 16 条独立累加链：FMA 延迟 4 周期 × 2 端口 = 至少 8 条链才能填满流水线。
- DRAM、L1、triad 取扫描两端各 3 个点的中位数：平台检测（相邻点下降超过 35% 才分级）会把 L2→L3 的缓坡并进一级，不适合直接取值。

## 白板估算

```text
depthwise 3×3，56×56×144，fp32：I ≈ 2.25 FLOP/B（输入输出各一次）
  屋顶 = 2.25 × 13.4 = 30 GFLOP/s；实测 NHWC SIMD 10.8 → 效率 36%
1×1 卷积 56×56，24→144，fp32：I = 10.2 FLOP/B，略低于拐点 11.5
  屋顶 = 137 GFLOP/s；实测最快 30.5 → 22%（教学 GEMM 没有寄存器分块与打包）
ResNet 3×3 14×14×256：I = 84 FLOP/B，远在拐点右边，屋顶 = 154.9
  implicit GEMM 34.9 → 23%；Winograd F(2,3) 等效 75.0 → 48%（乘法少了 2.25 倍）
int8 VNNI 打包 GEMM，square_512：I 远大于 29.5，屋顶 396 → 实测 212，效率 53%
0.5 B 参数模型的 decode（Q4_0 0.5625 B/参数 = 0.28 GB，只计权重）：
  本机 1 线程 0.28 / 13.4 = 21 ms/token；4 线程 5.8 ms；Pi 5 以总线峰值 17.1 计 16.5 ms（实际更慢）
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 带宽阶梯 | `cpp/roofline.cpp` `bandwidth`：16 KiB–512 MiB，只读求和与 triad，1/4 线程 | 每个点 11 次取中位数；`test_cpp_smoke` 编译运行 |
| fp32 / int8 峰值 | `fma_kernel`（16 链 `vfmadd`）、`int8_kernel`（16 链 `vpdpbusd`）；AVX2 回退 | ISA 写进每行 JSON；EDGE_MARCH=haswell 时走 AVX2 分支 |
| 平台检测、拐点、效率 | `code/hostroof/analysis.py`：`plateaus`、`ridge`、`attainable`、`efficiency` | `tests/test_roofline.py` 6 个测试 |
| 实测点上屋顶 | `scripts/run_experiments.py` `lab_points`：读 conv-kernels、int8-kernels 的 CSV | `lab_points.csv` 每点带强度与效率 |
| 线程池 | `edge_harness/include/edgebench.h` `ThreadPool` / `parallel_for` | `tools/tests` 空派发 < 100 µs、每个下标恰好一次 |

## 易错点与反例

- 用数据手册带宽画屋顶：Pi 5 的 17.1 GB/s 是总线算术；本机单核只读 13.4 GB/s，而 4 核能到 48.4，单核持续值和总线峰值之间可以差好几倍。
- 用 fp32 峰值 × 4 当 int8 峰值：本机实测 2.56 倍，拐点会被高估 1.6 倍。
- 测带宽时每次扫描都派发线程：4 线程 L1 读从 596 掉到约 50 GB/s，测到的是派发开销。
- 强度只按 DRAM 算却拿 L1 带宽做屋顶：两边要对应同一层存储。
- 把效率低归因于「受带宽限制」而不看强度：1×1 卷积 56×56 的强度 10.2 FLOP/B，屋顶 137 GFLOP/s，实测 30 GFLOP/s 离两条屋顶都很远，瓶颈是 kernel 本身。
- 忽略后台负载：建仓时一个遗留进程占了一个核 6 天，4 线程测量全部偏低；任何计时前先看一眼 CPU 占用。

## 实验或练习

- 假设：带宽随工作集呈阶梯；单核喂不满 DRAM；int8 峰值与 fp32 之比低于 4；计算受限的教学 kernel 离屋顶比带宽受限的远。
- 配置：`cpp/roofline.cpp`，-O3 -march=native，1 与 4 线程，工作集 16 KiB–512 MiB 每次翻倍，11 次取中位数；峰值核 2000 万次迭代 × 7 次。
- 结果：1 线程 L1 166 / L2 约 120 / L3 27–40 / DRAM 13.4 GB/s，4 线程 596 / 约 415–450 / 117–119 / 48.4；fp32 154.9 → 596.6 GFLOP/s，int8 396.1 → 1492.7 GOP/s；GEMM 类卷积效率 16–26%，depthwise NHWC 36–87%，朴素 NCHW 不到 1%。
- 练习：给 `sum_read` 加软件预取或把 4 个累加器换成 8 个，看单核 DRAM 读能否超过 13.4 GB/s；把 triad 改成 non-temporal store（`_mm512_stream_ps`），验证 write-allocate 的那一份流量消失。

## 高频问题

- 问：为什么 roofline 的横轴是对数？
  答：强度从 0.1（逐元素）到上千（大 GEMM）跨四个数量级，对数坐标下两条屋顶都是直线，拐点一眼可见。
- 问：L1 的屋顶什么时候有用？
  答：分块后数据驻留在 L1 时。1 线程 fp32 从 L1 供数的拐点只有 0.93 FLOP/B，几乎所有分块好的 kernel 都是计算受限；没分块时才看 DRAM 屋顶。
- 问：端侧 NPU 的 roofline 怎么画？
  答：算力用稠密 int8 峰值（厂商 TOPS 常含 2:4 稀疏，要先去掉），带宽用 DRAM 持续值与片上 SRAM 两层；另加一条「DMA 搬运」屋顶，见 M07。
- 问：一个点画在屋顶上方说明什么？
  答：强度算错了：通常是数据实际命中了更快的一层（工作集小于缓存），或者没算某个张量的流量。

## 面试追问链

1. 问：给一个算子，你怎么判断它受什么限制？
   答：先算 E3 的强度，再和目标芯片的拐点比。本机 fp32 拐点 11.5 FLOP/B：depthwise 2.25 在左边，ResNet 3×3 的 84 在右边，1×1 卷积 10.2 在拐点附近。
2. 问：拐点的两个数从哪来？
   答：实测。fp32 峰值用 16 条独立 FMA 链测（154.9 GFLOP/s），带宽用大于 L3 的只读扫描测（13.4 GB/s）。数据手册的峰值只用来检查测量是否合理，例如由峰值反推频率 2.42 GHz。
3. 问：为什么单核只有 13.4 GB/s，4 核却有 48.4？
   答：Little 定律（E2）：持续带宽 = 在途字节 / 延迟。每核能挂起的 miss 数是固定的，所以单核带宽是并发上限；4 核在途数 4 倍，带宽涨 3.6 倍，说明总线还没满。triad 按写分配计有 18.4 GB/s，也证明单核只读没占满总线。
4. 问：int8 比 fp32 快多少？
   答：峰值上 2.56 倍，不是 4 倍：VNNI 指令吞吐只有 FMA 的 64%。实际 kernel 还要看 requant 与布局，ORT 1 线程实测 1.4–3.1 倍（M02）。
5. 问：decode 为什么慢，怎么估？
   答：GEMV 强度 ≈ 2 / 每参数字节，Q4_0 约 3.6 op/B，远在拐点左边，时间 ≈ 权重字节 / 持续带宽。0.5 B 参数 Q4_0 在本机单线程 21 ms/token，4 线程 5.8 ms；实测见 M08。
6. 问：屋顶下面的点离屋顶很远，怎么办？
   答：看它在拐点哪侧：左边先减字节（融合、量化、换布局），右边先喂满 SIMD（寄存器分块、打包、去掉水平求和）。本机 1×1 卷积 30 GFLOP/s 离 137 的屋顶很远，缺的是寄存器分块，这是 M06 调度搜索要解决的。

## 自测题

1. 一颗 MCU 主频 480 MHz、每周期 2 个 int8 MAC，SRAM 总线 4 字节/周期，int8 拐点是多少？
   要点：算力 480e6 × 2 × 2 = 1.92 GOP/s；带宽 1.92 GB/s；拐点 1 op/B。
2. fp16 GEMV，N = K = 4096，强度是多少？
   要点：2·4096² / (2·4096² + 2·8192) ≈ 1.0 FLOP/B。
3. 本机 4 线程 fp32 GEMM 512×512×512，从 DRAM 每个矩阵只搬一次，强度与屋顶？
   要点：2·512³ / (4·3·512²) = 85 FLOP/B > 12.3，计算受限，屋顶 596.6 GFLOP/s。
4. 某芯片宣称 8 核 3 GHz、AVX-512 单端口、fp32 峰值 3 TFLOP/s，可信吗？
   要点：8 × 3e9 × 1 × 16 × 2 = 0.77 TFLOP/s，要 4 个端口才到 3 TFLOP/s，不可信。

## 关联笔记

- [M02 整数契约](module-02-integer-contract.md)：int8 峰值只有 fp32 的 2.56 倍，requant 怎么吃掉剩下的收益
- [M05 卷积 kernel](module-05-convolution-kernels.md)：本讲屋顶下的卷积实测点
- [M06 调度搜索](module-06-schedule-search.md)：离屋顶远的 GEMM 怎么靠搜索分块与循环顺序逼近
- [M08 分块量化 decode](module-08-block-quantized-decode.md)：GEMV 的带宽上界与实测
- [mlsys-infra M01 性能模型与 roofline](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-01-performance-models.md)：GPU 上的 roofline 与 β·L 在途字节推导，本讲给出 CPU 上的实测版本
- [mlsys-infra M03 并行编程与调度](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-03-parallel-programming-scheduling.md)：CS149 A2 实测的派发开销（自旋池 0.78 µs、睡眠池 10.8 µs、新建线程 53.3 µs），解释了为什么测带宽不能每次扫描都派发

## 本讲小结

屋顶要实测：本机单核 fp32 154.9 GFLOP/s、DRAM 13.4 GB/s，拐点 11.5 FLOP/B；int8 只有 fp32 的 2.56 倍。
带宽是阶梯且单核喂不满 DRAM：Little 定律决定单核持续带宽，4 线程 3.6 倍。
端侧 kernel 先算强度、找到它所在的屋顶，再决定是减字节还是喂满 SIMD；本仓库 GEMM 类卷积离屋顶还有 4–6 倍。

## 延伸阅读

- Williams, Waterman & Patterson, Roofline, CACM 2009, https://doi.org/10.1145/1498765.1498785
- McCalpin, STREAM: Sustainable Memory Bandwidth in High Performance Computers, https://www.cs.virginia.edu/stream/
- Ofenbeck et al., Applying the Roofline Model, ISPASS 2014, https://doi.org/10.1109/ISPASS.2014.6844463
- Intel 64 and IA-32 Architectures Optimization Reference Manual（Ice Lake 的 FMA 端口与 fill buffer）, https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
- uops.info（指令延迟与吞吐的实测表）, https://uops.info/
