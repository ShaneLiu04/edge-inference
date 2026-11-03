# 手写题

端侧部署、推理引擎、NPU 编译岗的手写题集中在下面这些。每题给出：限时、参考实现、能直接跑的测试、最容易错的地方。
练法：关掉参考实现，在空文件里写，写完跑对应测试；不过再看参考。数据中心侧的手写题（FlashAttention、ring all-reduce、分页 KV）在 [mlsys-infra CODING](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/interview/CODING.md)。

测试命令统一是在实验目录里 `python -m pytest -q tests/<文件> -k <关键字>`；`tools/` 下的在仓库根目录 `python -m pytest -q tools/tests/<文件>`。

## 一、整数契约（部署 / 量化岗必练）

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C1 | 把实数乘数 M 拆成 Q31 的 m0 和 shift | 10 min | [`requant.py`](../solutions/edge-runtime/int8-kernels/code/int8ref/requant.py) `quantize_multiplier` | `test_requant.py -k multiplier` | m0 舍入到 2^31 时要右移一位、shift 减一；M ≥ 1 时 shift 为负（左移） |
| C2 | SRDHM + RoundingDivideByPOT | 15 min | 同上 `srdhm`、`rounding_divide_by_pot` | `test_requant.py -k "srdhm or pot"` | 平局向 +∞ 不是远离零；负数右移的舍入要加 `(x < 0)` 修正；`INT32_MIN × INT32_MIN` 饱和 |
| C3 | int8 卷积：int32 累加 + 逐通道 requant + zero-point | 25 min | [`layer.py`](../solutions/edge-runtime/int8-kernels/code/int8ref/layer.py) `acc_int32`；C++ `int8.cpp` `conv_int32` | `test_cpp_and_ort.py -k bit_exact` | 输入 zero-point 要从累加里减掉 `zx · Σw`（可预计算进 bias）；padding 填的是 zx 不是 0 |
| C4 | BN 折叠进卷积 | 10 min | [`fold.py`](../solutions/mit6.5940/lab2-quantization/code/edgequant/fold.py) `fold_bn` | `test_fold.py -k fold_matches` | 用 `var + eps` 的平方根；没有 bias 的卷积折叠后会有 bias |
| C5 | 对称 per-channel / per-tensor 权重量化 | 10 min | 同上 `quantize_symmetric`；[`quant.py`](../solutions/mit6.5940/lab2-quantization/code/edgequant/quant.py) `fake_quant_weight` | `test_fold.py -k per_tensor`、`test_quant.py -k weight_fake` | 范围用 ±127 不是 ±128；全零通道 scale 为 0 要防除零 |
| C6 | 非对称激活量化参数 | 10 min | `quant.py` `act_qparams` | `test_quant.py -k act_qparams` | 范围必须包含 0，否则 0 不能精确表示（padding、ReLU 输出全错） |

## 二、校准与量化训练

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C7 | MSE 校准（网格搜索裁剪比例） | 15 min | `quant.py` `calib_mse` | `test_quant.py -k heavy_tail` | 误差要在量化-反量化之后算，不是只算裁剪误差 |
| C8 | KL（entropy）校准 | 30 min | `quant.py` `calib_kl`、`calib_kl_nonzero` | `test_quant.py -k kl` | 裁掉的尾部质量要加到参考分布的最后一个 bin；候选分布只在参考非空的 bin 上展开；ReLU 后的零值尖峰要排除 |
| C9 | STE 伪量化（带梯度裁剪） | 10 min | `quant.py` `STEQuant` | `test_quant.py -k ste` | 区间外的梯度是 0 不是 1 |
| C10 | SQNR 与逐层误差定位 | 10 min | `quant.py` `sqnr_db`；[`qsim.py`](../solutions/mit6.5940/lab2-quantization/code/edgequant/qsim.py) `point_outputs` | `test_quant.py -k sqnr` | 比较的是同一输入下的浮点与量化激活，不是两次独立前向 |

## 三、内存与调度（MCU / NPU 编译岗必练）

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C11 | 存活期下界 + 按大小贪心 arena | 25 min | [`planners.py`](../solutions/mit6.5940/tinyengine-memory/code/memplan/planners.py) `lower_bound`、`greedy_by_size`、`validate` | `test_memplan.py -k planners` | 区间端点包含；只与存活期相交的块比较地址 |
| C12 | first-fit 碎片反例 | 10 min | [`edge_calc.py`](../tools/edge_calc.py) `first_fit_arena` | `tools/tests/test_edge_calc.py -k fragment` | 反例要让「先占低地址的块先死」，留下放不下后续块的洞 |
| C13 | receptive field 回推（patch 推理的 halo） | 10 min | [`patch.py`](../solutions/mit6.5940/tinyengine-memory/code/memplan/patch.py) `input_extent` | `test_memplan.py -k receptive` | 从 cut 往输入方向逆序回推；1×1 卷积不改变范围 |
| C14 | 双缓冲 / 多级流水的事件模拟 | 25 min | [`pipeline.py`](../solutions/edge-runtime/dma-double-buffer/code/tiling/pipeline.py) `simulate` | `test_tiling.py` | 同一引擎上 tile 要按顺序；共享 DMA 时搬入和搬出串行；缓冲数限制第 i+B 块的搬入要等第 i 块算完 |
| C15 | L1 容量约束下的分块搜索 | 25 min | [`dory.py`](../solutions/edge-runtime/dma-double-buffer/code/tiling/dory.py) `search` | `test_tiling.py -k dory` | 双缓冲时输入、输出 tile 都要 ×2；每次 DMA 有固定启动开销，只看带宽会选出最小 tile |
| C16 | GEMM 调度空间与代码生成 | 30 min | [`schedule.py`](../solutions/mlc/tensor-program-tuning/code/tuner/schedule.py) `Space`、`render` | `test_tuner.py -k "space or render"` | tile 必须整除维度；循环顺序改变时边界与累加初始化的位置 |

## 四、算子（CPU 推理引擎岗必练）

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C17 | Winograd F(m, r) 的 Toom-Cook 构造 | 30 min | [`winograd.py`](../solutions/edge-runtime/conv-kernels/code/convlab/winograd.py) `transform` | `test_winograd.py -k "lavin or exact"` | 插值点个数 = m + r − 2 个有限点 + ∞；点选 0、±1、±2、±1/2 才能控制条件数 |
| C18 | im2col + GEMM 卷积 | 20 min | C++ [`conv.cpp`](../solutions/edge-runtime/conv-kernels/cpp/conv.cpp) `conv_im2col` | `test_winograd.py -k cpp` | 展开顺序要和权重 reshape 一致；工作区 OH·OW·KH·KW·IC |
| C19 | int8 GEMM 的 VNNI 打包布局 | 30 min | C++ [`int8.cpp`](../solutions/edge-runtime/int8-kernels/cpp/int8.cpp) `pack_vnni` | `test_cpp_and_ort.py -k kernels_bit_exact` | 布局 [N/16][K/4][16][4]：一条 `vpdpbusd` 处理 16 个输出列 × 4 个 K；u8 是激活、s8 是权重，不能反 |
| C20 | Q4_0 / Q8_0 打包与反量化 | 15 min | [`pack.py`](../solutions/edge-runtime/q4-gemv/code/blockq/pack.py) `quantize_q4_0`、`dequantize` | `test_blockq.py -k "gguf or largest"` | Q4_0 的 d = max / −8（带符号的最大值），使绝对值最大的元素精确；低 nibble 是前 16 个元素 |
| C21 | Q4_0 × Q8_0 整数点积 | 20 min | C++ [`gemv.cpp`](../solutions/edge-runtime/q4-gemv/cpp/gemv.cpp) `gemv_q4_q8_vnni` | `test_blockq.py -k cpp_kernels` | `dpbusd` 要 u8 × s8：nibble 0..15 当 u8，减 8 的偏移用 `8 · Σx` 一次扣掉 |
| C22 | roofline 平台识别 | 15 min | [`analysis.py`](../solutions/edge-runtime/host-roofline/code/hostroof/analysis.py) `plateaus`、`ridge` | `test_roofline.py` | 平台之间以带宽下降比例切分，噪声不能产生新平台 |

## 面试现场的顺序

1. 先说数据类型和布局（「激活 uint8、zero-point 128，权重 int8 对称、逐通道，累加 int32」），再写代码。
2. 先写浮点或 int64 的参考，再写定点或 SIMD 版本，测试用逐位相等，不用 allclose。
3. 写完主动说边界：溢出上界（127² · K）、平局舍入、zero-point 对 padding 的影响、不整除的尾块。
