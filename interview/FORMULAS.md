# 公式卡：E1–E27

编号与笔记一致（M01 的 E1–E3 …… M08 的 E25–E27），推导在对应笔记的「关键推导」。每条给一个口算、一个本仓库的实测锚点、一个常见错答。
带 `edge_calc` 的条目可以用 `python tools/edge_calc.py` 复算，测试在 `tools/tests/test_edge_calc.py`；实测数字由根目录 [`claims.json`](../claims.json) 校验。
数据中心一侧的同类公式（F1–F22，H100、HBM、NVLink）见 mlsys-infra [FORMULAS](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/interview/FORMULAS.md)。

本机：Ice Lake 服务器虚拟机，4 物理核，AVX-512 + VNNI。1 线程 fp32 154.9 GFLOP/s、int8 396.1 GOP/s、DRAM 只读 13.4 GB/s；4 线程 596.6 / 1492.7 / 48.4。

## M01 roofline（[笔记](../notes/module-01-edge-roofline.md)）

**E1 roofline 与拐点** · `edge_calc.ridge`

```text
P_att(I) = min(P_peak, I·β)，I* = P_peak / β
```

- 口算：本机 1 线程 fp32 154.9 / 13.4 = 11.5 FLOP/B，int8 396.1 / 13.4 = 29.5 op/B。
- 锚点：GEMM 类卷积在屋顶的 16–26%，depthwise NHWC 36–87%（它的屋顶是斜线），朴素 NCHW 不到 1%。
- 常见错答：用厂商 TOPS 当 P_peak。INT8 TOPS 常含 2:4 稀疏，稠密只有一半；本机 int8 按位宽应是 fp32 的 4 倍，实测 2.56 倍。

**E2 Little 定律**

```text
β_sustained = 在途字节 / 访存延迟 = 在途 cache line 数 × 64 B / latency
```

- 口算：延迟取 100 ns，13.4 GB/s 对应约 21 条 line 在途；4 核 48.4 GB/s 对应每核约 19 条，每核在途数不变。
- 锚点：同样是 1 线程，只读核 13.4、triad（按写分配计）18.4 GB/s；4 线程多行 fp32 GEMV 58.1 GB/s，高于同样工作集的只读核 52.7。访问流越多、在途越多，带宽越高，所以 13.4 是并发上限，不是总线上限。
- 常见错答：「单核带宽低是因为总线窄」。加线程有效、加 SIMD 宽度无效，说明受限的是在途请求数。

**E3 算术强度** · `edge_calc.gemv_intensity`

```text
GEMM  I = 2MNK / (b_a·MK + b_w·KN + b_c·MN)
GEMV  I → 2 / b_w（N、K 都大时）
```

- 口算：fp32 GEMV 0.5，fp16 1，int8 2，Q4_0 3.6 FLOP/B，全部远低于 11.5 的拐点。
- 锚点：M08 的 7B 形状 GEMV，fp32 单线程 4763 µs，换算 14.1 GB/s。
- 常见错答：int8 把算力减半所以延迟减半。GEMV 本来就受带宽限，减的是字节。

## M02 整数契约（[笔记](../notes/module-02-integer-contract.md)）

**E4 整数卷积** · `edge_calc.bias_int`

```text
acc[c] = Σ_k x_q[k]·w_q[k,c] − z_x·Σ_k w_q[k,c] + b_q[c]，b_q[c] = round(b[c] / (s_x·s_w[c]))
满幅上界 |acc| ≤ 255·127·KH·KW·IC（u8 × s8）
```

- 口算：3×3、IC = 256 时上界 7.5×10^7，int32 放得下；7×7、IC = 4096 时 6.5×10^9，放不下。
- 锚点：6 种层的 int32 累加 C++ 与 Python 参考 100% 相等，实际 |acc| 比上界小 5–49 倍。
- 常见错答：「int32 对任何卷积都够」。上界随归约长度线性涨，深层大卷积要拆部分和。

**E5 requant** · `edge_calc.requant_mul`

```text
y_q = clamp(z_y + round(M[c]·acc[c]), 0, 255)，M[c] = s_x·s_w[c] / s_y = M0·2^-(31+shift)
```

- 口算：M = 0.0123 时 M0 = round(0.0123·2^37)、shift = 6，M0 落在 [2^30, 2^31)。
- 锚点：浮点乘数、单步定点与 ORT `QLinearConv` 100% 一致。
- 常见错答：把 z_x 那一项留到运行时逐元素算。Σ_k w_q[k,c] 是常数，离线折进 bias。

**E6 两步舍入与 maddubs 饱和**

```text
两步舍入（SRDHM + rounding shift）不一致率 ≈ 2^-(shift+1)
vpmaddubsw 不饱和条件：2·255·max|w| ≤ 32767 ⇒ max|w| ≤ 64
```

- 口算：shift = 10 时约 0.05%。
- 锚点：TFLite 式两步舍入与 ORT 在 99.93–99.97% 的输出上一致，构造 .5 平局时 98.97–99.55%；|w| = 100 时 99% 的列出错，|w| = 127 时全错。
- 常见错答：「两个 runtime 对不上一个级，是 bug」。先查舍入步数；差 1 个级且比例与 2^-(shift+1) 吻合时是契约差异。

## M03 校准与误差定位（[笔记](../notes/module-03-calibration-localization.md)）

**E7 仿射量化与 BN 折叠** · `edge_calc.affine`、`edge_calc.bn_fold_scale`

```text
s = (hi − lo) / (2^b − 1)，z = round(−lo / s)，x̂ = s·(clamp(round(x/s) + z, 0, 2^b − 1) − z)
BN 折叠：inv = γ / sqrt(var + ε)，W' = W·inv，b' = (b − μ)·inv + β
```

- 口算：0.4 和 127 共用一个对称 scale（127/127 = 1）时 rint(0.4) = 0，小通道整个消失。
- 锚点：BN 折叠后 88.16% 不变，logits 最大差 4.9e-6；W4 逐张量 76.42%、逐通道 83.20%。
- 常见错答：「折叠是为了减小误差」。折叠是为了让整数图里没有 BN；折叠后通道幅值拉开，逐张量 scale 由最大通道决定。

**E8 裁剪权衡**

```text
MSE(hi) ≈ s²/12 · P(|x| ≤ hi) + E[(|x| − hi)² · 1{|x| > hi}]，s ∝ hi
```

- 口算：hi 减半，舍入噪声降为 1/4，裁剪项从 0 开始涨；最优点由尾部质量决定。
- 锚点：A8 上 minmax、percentile、MSE 三种校准差距 < 0.2 个点。
- 常见错答：「位宽够就不用关心校准」。A8 上确实差别小，A6、A4 时裁剪点决定结果。

**E9 KL 校准**

```text
threshold = argmin_i KL(P_i ‖ Q_i)，i 从 2^b 扫到 2048
```

- 口算：精确零在合并组里被摊薄，KL 多出约 p₀·ln(i/2^b)，只随截断点 i 增大，阈值被拉低到 MSE 的约 1/4。
- 锚点：TensorRT 式 KL 63.68%（A6 时 16.30%），去掉精确零后 88.06%。
- 常见错答：「KL 是最好的校准器」。它对直方图形状敏感，ReLU6 网络上要先去掉 0 bin。

## M04 静态内存规划（[笔记](../notes/module-04-static-memory-planning.md)）

**E10 激活峰值下界** · `edge_calc.peak_live`

```text
L = max_t Σ_{i: b_i ≤ t ≤ d_i} s_i
```

- 锚点：torchvision MobileNetV2 1.0@224 下界 1,505,280 B；每张量独占是下界的 4.7 倍。
- 常见错答：「把所有激活加起来就是峰值」。寿命不重叠的张量可以共用地址。

**E11 arena 与碎片率** · `edge_calc.first_fit_arena`

```text
A = max_i (off_i + s_i) ≥ L，碎片率 = A / L − 1
```

- 锚点：按大小降序贪心（TFLM `GreedyMemoryPlanner`）在 11 个配置上碎片率都是 0；按出生顺序 first-fit 13.3%。
- 常见错答：「贪心不是最优，所以一定有浪费」。下界可达时贪心经常就是最优；先算 L 再说浪费。

**E12 halo 回推**

```text
in = (out − 1)·stride + kernel，从 cut 处逐层往输入回推
```

- 口算：从输出 tile 7 往回推，先过一个 3×3 stride 2 层得 (7 − 1)·2 + 3 = 15，再过一个 3×3 stride 1 层得 (15 − 1) + 3 = 17。
- 常见错答：「patch 只多一圈 halo」。halo 随层深线性外扩，切得越深越大。

**E13 patch 重算**

```text
stage 开销 = p² · MACs(tile_in) / MACs(全图)
```

- 锚点：MBv2-1.0@224 切前 4 个 block、2×2 patch：stage 1.31 倍、整网 1.077 倍；0.35@160 切 8 个 block、4×4：整网 4.2 倍。
- 常见错答：「patch 越细越省」。峰值降得越来越慢，重算按 p² 涨。

**E14 patch 峰值**

```text
max(stage 输出整块 + 单 tile 在 stage 内的峰值, 其余层的峰值)
```

- 锚点：1.0@224 cut 4 p 2：1,505,280 → 566,784 B；stage 输出 56×56×24 = 75,264 B 那一项必须整块驻留。
- 常见错答：只算 tile 内的峰值，忘了 stage 输出和后半段网络。

## M05 卷积 kernel（[笔记](../notes/module-05-convolution-kernels.md)）

**E19 im2col 工作区** · `edge_calc.im2col_workspace`

```text
W = OH·OW·KH·KW·IC·b（1×1 stride 1 时退化为拷贝 H·W·IC·b）
```

- 口算：ResNet 56×56×64、3×3、fp32：56·56·9·64·4 = 7,225,344 B，是输入加输出的 4.5 倍。
- 锚点：implicit GEMM 工作区 0，速度 22.4–34.9 GFLOP/s，与 im2col 的 15.8–36.6 相当。
- 常见错答：「im2col 只是实现细节」。在几百 KB SRAM 上它经常比权重先撞上限。

**E20 Winograd F(m, r)**

```text
Y = Aᵀ[(G g Gᵀ) ⊙ (Bᵀ d B)]A，每 tile 乘法 (m + r − 1)²，直接卷积 m²r²
```

- 口算：F(2,3) 16 对 36，2.25 倍；F(4,3) 36 对 144，4 倍。
- 锚点：14×14×256 上 2.15 倍，56×56×64 上 1.16 倍；fp16 下 F(4,3) 误差是直接卷积的 7.2 倍，插值点换成 0..4 直接溢出。
- 常见错答：「乘法少 4 倍就快 4 倍」。变换开销和数值误差随 m 增长，端侧 fp16 / int8 基本只用 F(2,3)。

**E21 depthwise 强度** · `edge_calc.depthwise_intensity`

```text
I = 2·KH·KW / b（每个输入元素读一次，权重可忽略）
```

- 口算：fp32 3×3 4.5 FLOP/B，int8 18；滑窗不复用时再除以 9。
- 锚点：NHWC + SIMD 10.4–11.4 GFLOP/s，按通道 CHW 1.2–1.3（8.2–9.0 倍）；倒残差块里 depthwise 占 MACs 15.8%、时间 33.4%。
- 常见错答：「参数少所以快」。参数少只说明权重流量小，激活流量还在。

## M06 调度搜索（[笔记](../notes/module-06-schedule-search.md)）

**E22 空间大小**

```text
|S| = 6·6·|TI(M)|·|TJ(N)|·|TK(K)|·2·2
```

- 口算：256³：36 × 7 × 5 × 6 × 4 = 30,240；pw_expand_14（196×576×96）：3,888。
- 锚点：grid 在 expand_14 上只有 27 个点，64 次预算测完，三个种子结果相同。
- 常见错答：「空间大才需要搜索」。空间小、点之间差 7 倍时一样要搜，问题是测量噪声。

**E23 分块足迹与寄存器块**

```text
bytes = 4·(ti·tk + tk·tj + ti·tj) ≤ L1（48 KiB）/ L2（1.25 MiB）
累加器 = ti·tj / 8，要求 8 ≤ 累加器 ≤ 16（FMA 延迟 4 × 2 端口；16 个 ymm）
```

- 口算：t2×64×64：2·64 / 8 = 16 个累加器，每读一个 B 向量做 2 次 FMA。
- 锚点：最好的调度汇编里 32 条 ymm FMA；最内层 i 时 ti ≤ 4 中位 52.9、ti > 4 中位 10.9 GFLOP/s。
- 常见错答：「分块是为了 L1」。256³ 三个矩阵合计 768 KiB，本来就在 L2；决定性能的是寄存器块。

**E24 排序一致性**

```text
Spearman ρ = 1 − 6 Σ d_i² / (n (n² − 1))
```

- 锚点：岭回归代价模型留出 ρ = 0.47 / 0.53；64 次预算下中位最优 69.0 / 77.4，不如随机的 77.7 / 80.8。
- 常见错答：「有代价模型就比随机好」。ρ 只有 0.5 时，按模型排的前几名和随机抽差不多。

## M07 DMA 与双缓冲（[笔记](../notes/module-07-dma-double-buffering.md)）

**E15 单缓冲与双缓冲** · `edge_calc.single_buffer_time`、`edge_calc.pipeline_time`

```text
T1 = n·(m + c)，T2 = n·max(m, c) + min(m, c)
上限 T1 / T2 → (m + c) / max(m, c) ≤ 2
```

- 口算：8 块、m = c = 2：32 对 18；m = 5、c = 1、4 块：24 对 21。
- 锚点：depthwise 从 HyperRAM 供数效率 0.257，双缓冲 1.24 倍，封顶于 (m + c)/m ≈ 1.25。
- 常见错答：「双缓冲延迟减半」。只有 m ≈ c 且 tile 多时才接近 2 倍；省掉的是较短的那一侧。

**E16 k 级流水**

```text
T = Σ s_i + (n − 1)·max s_i（每级一个引擎、缓冲足够）
```

- 锚点：搬入、搬出共用一个 DMA 通道时三级退化成两级，DMA 侧时间是 m_in + m_out。
- 常见错答：「三级流水一定比两级快」。级数受独立引擎数限制。同形公式是 mlsys-infra M13 的流水线气泡。

**E17 tile 的 L1 占用**

```text
B·(in + w) + out，B 为缓冲数；切输入通道时 out 是 int32 部分和，只留一份
```

- 常见错答：双缓冲时输出也乘 2。部分和留在 L1 不搬，只有最后一块写回。

**E18 每 tile 时间**

```text
m = bytes / (B/cycle) + setup_dma，c = MACs / (MAC/cycle) + setup_call
```

- 锚点：本机两线程实测双缓冲 0.61–1.23 倍；4 KiB tile 时每块一次跨线程握手（约 0.8 µs）比搬运还长。
- 常见错答：忽略 setup。开销项让最优 tile 不是最小的那个。

## M08 块量化解码（[笔记](../notes/module-08-block-quantized-decode.md)）

**E25 块格式字节与解码上界** · `edge_calc.qblock_bytes_per_param`

```text
bpw = 块字节 × 8 / 32：Q8_0 34 B → 8.5，Q4_0 18 B → 4.5，Q4_1 20 B → 5.0
tok/s ≤ BW / (N_params × bpw / 8)
```

- 口算：Llama-7B（6.74×10^9）Q4_0 3.79 GB：Pi 5（17 GB/s）4.5 tok/s，Orin Nano（68 GB/s）17.9 tok/s。
- 锚点：7B 形状 4096² 单线程 Q4 × Q8 1034 µs，只有 9.1 GB/s；比 fp32 快 4.6 倍，字节比是 7.1 倍。
- 常见错答：「4 bit 是 fp16 的 1/4」。每 32 个数还有一个 fp16 scale，字节比 4.5 / 16 = 0.281。

**E26 Q4 × Q8 块点积**

```text
Σ_j d_w (q_j − 8)·d_x s_j = d_w d_x (Σ_j q_j s_j − 8 Σ_j s_j)
q_j ∈ [0, 15] 当 u8，s_j ∈ [−127, 127] 当 s8
```

- 锚点：整数点积比浮点融合快 1.7 倍（1034 对 1743 µs），先反量化再点积 4541 µs、和 fp32 差不多。
- 常见错答：「VNNI 两边都可以有符号」。dpbusd 一边必须是 u8；Q8 × Q8 时用 |w| 当 u8、sign(w)·x 当 s8。

**E27 量化噪声**

```text
误差功率 d²/12，d ∝ 块内最大值 / 2^(b−1)，SQNR(b) ≈ 6.02·b + C
```

- 口算：8 bit 到 4 bit 少 24 dB。
- 锚点：GPT-2 各层 Q8_0 45.08 dB、Q4_0 21.03 dB，差 24.04 dB；但 Q4_1 SQNR 更高（21.96）、困惑度更差（32.15 对 31.88）。
- 常见错答：「SQNR 高的格式困惑度一定低」。层平均 SQNR 低估了少数大幅值权重的贡献。

## 默写顺序

部署 / 量化岗：E4、E5、E7、E9、E25。MCU / NPU 岗：E1、E10、E14、E15、E21。CPU 推理引擎岗：E2、E3、E19、E20、E23。端侧 LLM：E3、E25、E26、E27，加 mlsys-infra 的 F15（KV cache）。
