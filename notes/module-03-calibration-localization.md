# Module 03 — 校准与误差定位：粒度、校准器、逐层敏感度、混合精度与 QAT

- 课程出处：MIT 6.5940 Fall 2024《Quantization I / II》Lab 2 · TensorRT《8-bit Inference with TensorRT》(GTC 2017) · ONNX Runtime `quantization` 文档
- 必读论文：Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, 2018；Krishnamoorthi, Quantizing deep convolutional networks for efficient inference: A whitepaper, 2018；Nagel et al., A White Paper on Neural Network Quantization, 2021
- 对应实验：[lab2-quantization](../solutions/mit6.5940/lab2-quantization/README.md)
- 实测结论：CIFAR-10 上 20.9 万参数的 MobileNetV2 式网络 fp32 88.16%；W8A8 用 minmax/percentile/MSE 校准都在 87.97–88.17%，TensorRT 式 KL 只有 63.68%（A6 时 16.3%），去掉精确零后的 KL 回到 88.06%；W4 逐通道 83.20%，逐张量 76.42%；按敏感度升 8 层到 W8（+9.5 KiB）回到 87.69%，3 个 epoch 的 QAT 回到 87.63%；ORT 1.31 的 Entropy 默认参数与 MinMax 完全相同，改成 2048 bin 真正搜索后反而掉到 87.17%
- 状态：已验证（数字来自 `artifacts/raw/summary.json` 与各 CSV，`claims.json` 校验）

## 摘要

M02 规定了整数图怎么算；这一讲回答「scale 从哪来、掉点了去哪找」。
权重的 scale 由权重本身决定，问题只是粒度（逐张量还是逐通道）。激活的 scale 必须从校准数据里估计，问题是裁剪：范围取大了分辨率低，取小了饱和。
四种校准器在 W8A8 上差距不到 0.2 个点，真正的风险在算法的隐含假设：TensorRT 式 KL 把 ReLU 输出里大量的精确零当成要拟合的分布，结果把范围裁到 MSE 最优值的四分之一左右，掉 24 个点。
掉点之后的定位顺序是：先分开权重与激活（只量化一边），再逐层只量化一层看敏感度，再沿深度看每个激活点的 SQNR，最后决定哪几层升精度或者做 QAT。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 权重对称、逐输出通道；激活非对称、逐张量。ReLU6 输出非负，零点为 0 时整个 uint8 范围都用在 [0, hi]。
2. 本仓库的网络 W8 时逐张量和逐通道几乎一样（88.15 vs 88.16），W4 时差 6.8 个点（76.42 vs 83.20）。粒度的代价随位宽放大。
3. 校准器排序在 A8 上几乎无差别，A6 上开始拉开：percentile 88.00、MSE 87.94、minmax 87.62。KL 是例外，它在两种位宽上都失败。
4. 敏感度不可加：26 层单独 W4 的掉点之和 4.23 个点，全部同时 W4 掉 4.96 个点；混合精度要按敏感度排序，不能按参数量。
5. QAT 是最后手段：3 个 epoch（每个约 110 s，4 核 CPU）把 W4A8 从 83.04 拉回 87.63；同样的恢复用混合精度只要多 9.5 KiB 权重。

**必须能默写的公式**

```text
E7  仿射量化：  s = (hi − lo) / (2^b − 1)，z = round(−lo / s)，x̂ = s·(clamp(round(x/s) + z, 0, 2^b − 1) − z)
    对称权重：  s_c = max|W_c| / (2^(b−1) − 1)，逐通道时 c 是输出通道
E8  裁剪权衡：  MSE(hi) ≈ s²/12 · P(|x| ≤ hi) + E[(|x| − hi)² · 1{|x| > hi}]，s ∝ hi
    SQNR(dB) = 10·log10(Σx² / Σ(x − x̂)²)，满幅正弦输入时 6.02·b + 1.76 dB，每少 1 bit 少 6 dB
E9  KL 校准：  threshold = argmin_i KL(P_i ‖ Q_i)，P_i 是前 i 个 bin 加上尾部，Q_i 是把 P_i 合并成 2^b 组再摊回
```

**一句话区分**

- 逐张量 vs 逐通道：一个 scale 由最大的通道决定 vs 每个输出通道一个；后者在整数 kernel 里只是 requant 乘数变成向量，没有额外的乘加。
- 校准 vs QAT：只改 scale vs 连权重一起改；前者几秒，后者要训练数据和若干 epoch。
- 敏感度 vs SQNR：只量化一层时的精度下降（任务指标）vs 某个张量与 fp32 的信噪比（信号指标）；SQNR 告诉你误差在哪进来，敏感度告诉你它是否要紧。

## 学习目标

- 写出 E7 的量化与反量化，说清非对称激活为什么让 ReLU6 输出不浪费半个范围；
- 用实测解释逐通道在什么位宽、什么层（depthwise）上必要；
- 实现 minmax、percentile、MSE、KL 四种校准器，构造一个让 KL 失败的分布并说清原因；
- 给一个掉点的量化模型，按「权重/激活分离 → 逐层敏感度 → 沿深度 SQNR」的顺序定位到层；
- 在权重字节预算下给出混合精度方案，并和 QAT 比较代价。

## 先修知识

- M02：整数契约、requant 乘数、BN 折叠；
- M01：权重字节决定 Flash 占用和解码时的带宽；
- mlsys-infra M06：训练侧的 INT8/FP8 与 per-channel scale，作为本讲推理侧的对照。

## 核心概念

### 1. 量化点放在哪

整数 runtime 真正存在内存里的张量才是量化点：网络输入、每个 conv-BN-ReLU6 块的输出（BN 已折叠、ReLU6 是 requant 的 clamp）、每个残差 add 的输出、全局池化的输出。
`qsim.py` 按这个规则在 26 个权重层、31 个激活点上插 fake quant。点放多了（比如 conv 和 ReLU 之间也量化）会高估误差，放少了会低估。

### 2. 粒度

折叠 BN 后，各输出通道的权重幅值被 γ/σ 拉开。`fold_toy.py` 的随机例子里最大与最小通道的 max|W| 相差 479 倍，逐张量 W8 时 15 个通道的相对误差超过 10%、最差的通道被整个量化成 0。
真实训练出来的网络温和得多：本仓库 26 层里通道幅值比最大是 8.2（stem），depthwise 层在 4.8–6.9。所以 W8 逐张量只掉 0.01 个点；但 W4 时只有 7 个正量化级，同样的比例意味着小通道只剩 1–2 个级，掉 6.8 个点。
只让 depthwise 用逐张量（其余逐通道）时 W4 是 82.06%，介于两者之间：depthwise 每个通道只有 9 个权重，通道间差异最大，对粒度最敏感。

### 3. 四种校准器

- minmax：取样本的 [min, max]，不裁剪；
- percentile：取 99.99 分位，裁掉极少数尾部；
- MSE：把 [min, max] 按同一比例收缩，100 个候选里取量化 MSE 最小的（E8）；
- KL：TensorRT 的熵校准，|x| 做 2048 bin 直方图，对每个候选截断点 i 比较截断分布和量化后分布的 KL 散度（E9）。

`calib_size.csv` 显示样本数从 8 张图到 2048 张图，minmax 与 MSE 的 A6 精度变化都不到 0.15 个点（每档 3 个随机种子）。这个网络的激活范围在 8 张图里就已经稳定，校准集大小不是瓶颈。

### 4. 误差定位的三把尺子

- 权重/激活分离：只量化权重（`a_bits=None`）或只量化激活，先知道是哪一边；
- 逐层敏感度：其余全部 fp32，只把一层 W4（或一个点 A4），看掉多少；
- 沿深度的 SQNR：全网同时量化，在每个激活点和 fp32 参考比 SQNR，看误差从哪一层开始累积。

### 5. 混合精度与 QAT

混合精度：从全 W4A8 出发，按某个顺序把层升回 W8，画「权重字节 vs 精度」。
QAT：前向用 fake quant，反向用 STE（直通估计：clip 范围内梯度为 1，范围外为 0），在折叠后的网络上继续训练，激活范围固定为校准值。

## 关键推导

### 非对称激活为什么对 ReLU6 重要

ReLU6 输出在 [0, 6]。对称量化强制 [−hi, hi]，uint8 的一半码字永远不会出现，等效少 1 bit，SQNR 少约 6 dB。
非对称量化 E7 取 lo = 0，z = 0，256 个码字全部落在 [0, 6]，分辨率 6/255 ≈ 0.0235。这也是 TFLite 与 CMSIS-NN 激活用 int8 加零点、权重用对称的原因：零点只在激活一侧出现，整数 kernel 里它变成一个可预先算好的偏置修正项（M02）。

### 粒度代价为什么随位宽放大

逐张量时通道 c 的有效量化级数是 (2^(b−1) − 1) · max|W_c| / max|W|。幅值比 r 的通道：

```text
W8：127 / r 级；r = 8.2 时 15 级，相对误差约 1/(2·15) ≈ 3%
W4：  7 / r 级；r = 8.2 时不到 1 级，整个通道被量化成 0 或 ±1 个级
```

实测 W8 逐张量最差通道的相对误差 4.1%（`weight_ranges.csv`，stem），与估算一致；W4 时同一层不可用。

### 敏感度为什么不可加

单层敏感度是在「其余层都精确」的前提下测的。多层同时量化时，误差沿深度传播、在残差 add 处叠加，下游层的输入已经带噪声，再加自己的量化噪声。
实测：26 层单独 W4 的掉点之和 4.23，全部 W4 一起掉 4.96，超出 17%。按敏感度排序仍然有效（前几名就是该保的层），但「把掉点之和控制在 X 以内」这种预算是乐观的。

## 深入推导

### KL 校准在 ReLU 输出上为什么失败

E9 的候选截断点 i 从 2^b（256，或 A6 时的 64）扫到 2048。对每个 i，参考分布 P 是前 i 个 bin（尾部质量加到第 i 个 bin 上），候选分布 Q 是把这 i 个 bin 每 i/2^b 个合并成一组、组内质量平均摊回参考里非空的 bin。

ReLU6 输出有大量精确零，全部落在第 0 个 bin，质量为 p₀。第 0 组包含 bin 0 到 bin k−1（k = i/2^b），如果这些 bin 里其他 bin 非空，Q 在 bin 0 上只剩约 (p₀ + 少量)/k：

```text
KL 中 bin 0 的贡献 ≈ p₀ · ln(p₀ / q₀) ≈ p₀ · ln k = p₀ · ln(i / 2^b)
```

这一项只随 i 单调增大，与真实的量化误差无关：零点 z 能精确表示 0，零根本不产生误差。p₀ 大的时候它压过尾部裁剪的代价，KL 一路偏向小的 i，也就是小的截断点。
搜索下限 i ≥ 2^b 决定了最多能裁多狠：A8 时最小截断是 amax 的 256/2048 = 1/8，A6 时是 64/2048 = 1/32，所以 A6 掉得更厉害（16.3%）。
修正很直接：在非零值上跑同一个搜索（`calib_kl_nonzero`），零交给零点。W8A8 回到 88.06%，A6 87.97%，与 MSE 持平。

`kl_points.csv` 逐点验证了这个解释（31 个激活点，每次只量化一个点到 A8，其余 fp32）：

```text
非负点（ReLU6 输出、池化）18 个：KL 截断 / MSE 截断 中位数 0.268，15 个点在 0.236–0.287
                               非零 KL / MSE           中位数 0.733，范围 0.667–1.0
有符号点（project 输出、残差 add）：KL / MSE ≈ 1.2，只量化该点时精度不变
单点掉点最大 6.96 个点（features.2.body.1，第二个 block 的 depthwise 输出）
31 个单点掉点之和 23.75，全部同时 KL 掉 24.48：激活截断的损失近似可加
```

失败只出现在非负点上，有符号点上 KL 反而比 MSE 保守，这正是零尖峰机制的预测：有符号张量的精确零很少。

### ORT 的 Entropy 为什么和 MinMax 一模一样

ONNX Runtime 1.31 的 `create_calibrator` 对 Entropy 默认 `num_bins = 128`、`num_quantized_bins = 128`。它的阈值搜索从 `num_quantized_bins // 2` 扫到 `num_bins // 2`，即 `range(64, 65)`，只有一个候选，且就是全范围。
所以默认 Entropy 产生的模型与 MinMax 字节相同（都是 318.53 KiB、精度 88.09%）。`quantize_static` 的 `extra_options` 只转发 5 个 `Calib*` 键，传 `num_bins` 会被静默忽略；要跑真正的熵搜索，得直接调 `create_calibrator(..., extra_options={"num_bins": 2048})`，把 `compute_data()` 的结果存成 calibration cache 再交给 `quantize_static`。实验里的 `Entropy 2048 bins` 一行就是这样做的。

结果是 87.17%，比 MinMax 的 88.09% 低 0.92 个点。方向与本仓库的 KL 一致，幅度小得多：ORT 的直方图是对称的 [−t, t]（非负张量有一半 bin 为空），每次候选扩展两侧各一个 bin，量化 bin 数固定为 128，零尖峰被摊薄的程度不同。结论不变：熵校准用在 ReLU 输出上要先验证，不能当作默认更好的选项。

### QAT 恢复了什么

W4A8 PTQ 83.04%，QAT 第 1 个 epoch 就到 87.49%，之后 87.50、87.63。大部分恢复发生在第一个 epoch：STE 让权重在量化格点附近重新排列，补偿的主要是逐通道舍入的系统偏差，而不是重新学特征。
学习率 2e-3（原训练的 1/50 量级）加余弦退火，不改激活范围。如果把激活范围也设成可学习（LSQ/PACT），A4 才值得做；本实验只做 W4。

## 白板估算

```text
本网络：209,354 参数，8.57 M MACs/图，26 个权重层，31 个激活点
权重存储（只计权重，bias 另算）：W8 197.2 KiB；全 W4 98.6 KiB
  按敏感度升 4 层 → 102.9 KiB，87.05%     升 8 层 → 108.1 KiB，87.69%
  按参数量升 8 层 → 176.6 KiB，83.62%（升的全是大而不敏感的 1×1 层）
fp32 → ORT int8 QDQ：823.7 KiB → 318.5 KiB（2.59 倍），1 线程 batch 1：372.4 µs → 253.9–270.7 µs（1.38–1.47 倍，7 组 × 200 次的组中位数再取中位数）
```

SQNR 的量级：8 bit 满幅正弦的理论值 49.9 dB，实测单个激活点 26.1–42.9 dB（W8A8 逐通道，全网同时量化的累积结果；stem 输出最高，随深度逐渐降到 26–29 dB）；logits 30.5 dB 时精度只掉 0.19 个点；W4A8 的 logits 9.3 dB 时掉 5.1 个点。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 量化原语与 STE | `code/edgequant/quant.py` | `test_quant.py`：往返误差 ≤ s/2，STE 在范围外梯度为 0 |
| 四种校准器与 kl_nonzero | `quant.CALIBRATORS` | `test_zero_spike_drags_kl_threshold_down`：60% 零时 KL < 0.4·MSE，非零 KL > 0.7·MSE |
| BN 折叠与量化点 | `code/edgequant/qsim.py` | 折叠前后 logits 最大差 4.9e-6 |
| 敏感度、定位、混合精度、QAT | `scripts/run_experiments.py` 的 `sensitivity` / `localize` / `mixed` / `qat` | 每步一个 CSV，`summary.json` 汇总 |
| ORT 对照 | `step_ort` | 三种 ORT 校准器加 2048 bin 的 Entropy |
| 图 | `artifacts/figures/{weight_ranges,localize,mixed_precision,kl_points}.png` | 每张图一个结论 |

## 易错点与反例

- 只看 W8A8 就说「校准器无所谓」：A6 时 minmax 比 percentile 低 0.38 个点，KL 在两种位宽都崩。
- 在含大量零的张量上直接跑 KL：零由零点精确表示，不该参与分布拟合。
- 以为 ORT 的 Entropy 就是 TensorRT 的熵校准：默认参数下它退化成 MinMax。
- 用参数量排序做混合精度：升 8 个最大的层多花 78 KiB 只回到 83.62%，按敏感度升 8 层只多 9.5 KiB 就到 87.69%。
- 把 SQNR 的绝对值当门槛：W8A8 的激活点只有 26–43 dB，远低于 50 dB 的理论值，但精度几乎不掉；要看的是相对 W8A8 基线从哪一层开始陡降。
- depthwise 逐张量：每通道 9 个权重、通道间幅值差最大，W4 时比全逐通道多掉 1.1 个点。

## 实验或练习

- 假设：校准器选择在 A8 上不重要，在 A6 上重要；KL 在 ReLU 输出上会过度裁剪。
- 配置：W8 逐通道 + A8/A6，5 种校准器，512 张校准图，10,000 张测试图。
- 结果：A8 时 minmax 87.97、percentile 88.17、MSE 88.13、kl_nonzero 88.06，KL 63.68；A6 时 87.62 / 88.00 / 87.94 / 87.97，KL 16.30。
- 练习：在 `kl_points.csv` 里找出 KL 截断点最小的激活点，只对它换用 MSE，看能恢复多少；再把 `calib_kl` 的直方图下限改成第一个非零 bin，与 `kl_nonzero` 比较。

## 高频问题

- 问：为什么权重用对称、激活用非对称？
  答：权重近似零均值对称分布，对称量化省掉零点，整数乘加里不用减零点；激活经过 ReLU 后非负，对称会浪费一半码字。激活的零点修正可以折进偏置。
- 问：逐通道量化在推理时有额外代价吗？
  答：整数乘加没有。只是 requant 的乘数与移位从标量变成长度为输出通道数的数组，每个输出元素按通道取一次。
- 问：校准集要多大？
  答：本实验 8 张图到 2048 张图之间精度变化不到 0.15 个点。先测一下范围随样本数的收敛，再决定；常见的 100–1000 张通常够，分布偏移（不同场景、光照）比样本数更要紧。
- 问：PTQ 掉点多少该上 QAT？
  答：先做混合精度。本实验 W4A8 掉 5.1 个点，升 8 层到 W8（多 9.5 KiB）就回到 0.5 个点以内；只有权重预算不允许升层时 QAT 才划算。

## 面试追问链

1. 问：你的 int8 模型掉了 3 个点，怎么查？
   答：先只量化权重、再只量化激活，分开两边；然后逐层单独量化测敏感度；再全网量化后沿深度看每个激活点的 SQNR（E8），找到误差陡增的那一层。本实验 W4A8 的 SQNR 在第一个 block 的 depthwise 输出就跌破 20 dB（17.3），紧接着的 project 降到 11.8，残差相加后又回到 20.6（跳连那一路只有激活量化误差）；此后再没回到 20 dB 以上。
2. 问：定位到是激活的问题，下一步？
   答：换校准器（MSE 或 percentile），检查是不是 KL 这类有隐含假设的算法；看该点的分布有没有大量零或长尾。本实验 KL 把 ReLU6 输出裁到 MSE 截断的约四分之一，换成非零 KL 后回到 88.06%。
3. 问：KL 为什么会裁那么狠？
   答：E9 里第 0 个 bin 的精确零被合并组摊薄，KL 多出约 p₀·ln(i/2^b) 一项，只随截断点增大；这项与真实误差无关，因为零点精确表示 0。A6 的搜索下限更低（amax 的 1/32），所以崩得更厉害。
4. 问：别的工具的熵校准也有这个问题吗？
   答：不能假定没有。本仓库在 ORT 上用 2048 bin 的真实熵搜索复现了同一方向的损失（87.17% vs MinMax 88.09%），幅度小是因为它的直方图与合并方式不同。任何 KL 实现都要先看它怎么处理第 0 个 bin，再在 ReLU 输出上逐点比较截断点（`kl_points.csv` 的做法）。
5. 问：是权重的问题呢？
   答：看粒度和层类型。W4 逐张量比逐通道多掉 6.8 个点，depthwise 最敏感；最敏感的单层是 features.2.body.2.0（第二个 block 的 project 1×1，W4 时掉 0.71 个点）。按敏感度把前几层升 W8，权重只多几 KiB。
6. 问：混合精度方案怎么给出来？
   答：按单层敏感度降序升精度，画权重字节 vs 精度的曲线，在预算点上取。注意敏感度不可加（单独之和 4.23 vs 一起 4.96），曲线要实测，不能用加和预测。
7. 问：最后还差 0.5 个点，要不要 QAT？
   答：看代价。QAT 需要训练数据和管线，3 个 epoch 在 4 核 CPU 上约 5.5 分钟，W4A8 回到 87.63%；如果 Flash 还有 10 KiB 余量，升层更便宜、可复现。

## 自测题

1. 激活范围 [0, 6]，uint8 非对称，scale 和零点是多少？3.0 量化成多少？
   要点：s = 6/255 ≈ 0.02353，z = 0，round(3/0.02353) = 128。
2. 某层逐通道幅值比 20，W4 逐张量时最小的通道还有几个正量化级？
   要点：7/20 < 1，整个通道被量化成 0 或 ±1 个级，实际不可用。
3. 一个张量 70% 是精确零，A8 KL 搜索，截断点 i = 1024 时 bin 0 的额外 KL 大约多少？i = 256 呢？
   要点：0.7·ln(1024/256) = 0.97 nat；i = 256 时 k = 1，没有这一项。
4. 单层 W4 敏感度之和 4.2 个点，预算是掉 4 个点以内，全 W4 可行吗？
   要点：不能直接判断，敏感度不可加；本实验全 W4 实测 4.96，超出预算。

## 关联笔记

- [M02 整数契约](module-02-integer-contract.md)：scale、零点与 requant 乘数在整数 kernel 里怎么落地
- [M08 块量化与解码](module-08-block-quantized-decode.md)：块内共享 scale 是逐通道再细一级的粒度
- [M01 端侧 roofline](module-01-edge-roofline.md)：W4 省下的是权重字节，只在带宽受限时转化为速度
- [mlsys-infra M14 显存优化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-14-memory-optimization.md)：训练侧混合精度的 loss scale 与 FP8 per-tensor scale 的对照
- [mlsys-infra M17 压缩与量化](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)：服务端 INT8/INT4 权重量化（GPTQ/AWQ）与本讲的校准方法

## 本讲小结

权重的问题是粒度，激活的问题是裁剪；W8A8 时两者都很宽容，掉点出现在 W4 或 A6 以下。
校准器之间的差别小于它们隐含假设带来的风险：KL 在 ReLU 输出上把范围裁到四分之一，ORT 的默认 Entropy 根本不搜索。
定位按「分离 → 逐层敏感度 → 沿深度 SQNR」走，修复按「换校准器 → 按敏感度混合精度 → QAT」的代价顺序走。

## 延伸阅读

- Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, CVPR 2018, https://arxiv.org/abs/1712.05877
- Krishnamoorthi, Quantizing deep convolutional networks for efficient inference: A whitepaper, 2018, https://arxiv.org/abs/1806.08342
- Nagel et al., A White Paper on Neural Network Quantization, 2021, https://arxiv.org/abs/2106.08295
- Migacz, 8-bit Inference with TensorRT, GTC 2017, https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
- ONNX Runtime quantization, https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
