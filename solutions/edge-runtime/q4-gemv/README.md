# q4-gemv：llama.cpp 块量化的 GEMV 与 GPT-2 困惑度

对应笔记 [M08 块量化解码](../../../notes/module-08-block-quantized-decode.md)。端侧 LLM 解码每个 token 都要把权重读一遍，batch 1 的 GEMV 被带宽限住；
llama.cpp 的 Q8_0 / Q4_0 / Q4_1 把每 32 个权重和一个 fp16 scale 打成一块来减字节。本实验：

1. 用 NumPy 实现三种块格式，与 gguf 参考逐字节一致（`code/blockq/pack.py`）；
2. 用 C++ 写 9 种 GEMV（fp32、fp16、Q8 / Q4 的浮点融合、先反量化、VNNI 整数点积及其 4 累加器版），在 4 个 LLM 层形状上测 1 / 4 线程；
3. 不依赖 transformers 实现 GPT-2 前向（`code/blockq/gpt2.py`），在 wikitext-2 全部 280 个窗口上测困惑度，协议与 mlsys-infra 相同。

数据中心一侧的权重量化（GPTQ、AWQ、SmoothQuant，同一 GPT-2 协议）见 mlsys-infra [compiler-compression](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/solutions/cmu15849/compiler-compression/README.md) 与 [M17](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-17-compression-quantization.md)。

## 一句话结论

- **Q4 × Q8 VNNI 单线程比 fp32 快 4.6 倍，字节少 7.1 倍**：7B 形状 4096² 上 4763 → 1034 µs，实际只有 9.1 GB/s，被每块的解包指令限住。
- **先反量化再点积没有收益**：4541 µs，与 fp32 的 4763 差不多。
- **Q8 单线程与 Q4 持平，4 线程反超**：1009 vs 1034 µs，275 vs 311 µs；但 Q8 测到 64.7 GB/s，超过 DRAM 流式 48.4 GB/s，有 L3 命中，整模型流式读时结论很可能反转。
- **多累加器不是答案**：4 个独立累加器让 Q4 只快 4%、Q8 快 11%（4 线程时 Q8 反而慢 8%），Q4 的瓶颈是每块的指令吞吐，不是 FMA 依赖链。
- **GPT-2 124M 困惑度**：fp32 29.951、Q8_0 29.953、Q4_0 31.88、Q4_1 32.15；Q4_1 的层平均 SQNR 更高（21.96 vs 21.03 dB）困惑度却更差。mlsys-infra 同协议 W4 g128：RTN 33.27，GPTQ 31.20。

## 怎么跑

```bash
cd solutions/edge-runtime/q4-gemv
python -m pytest -q tests
python scripts/run_experiments.py --only bench   # 计时，机器空闲，约 2 分钟
python scripts/run_experiments.py --only gpt2    # 需要 .data/hf/models/gpt2 与 wikitext-2，4 种格式约 18 分钟
python scripts/run_experiments.py --only         # 只用已有原始结果重建 summary.json 与 decode_bound.csv
```

GPT-2 权重和 wikitext-2 通过 hf-mirror 下载到 `<repo>/.data/hf/`，不入库。`--windows N` 可只评测 N 个随机窗口（默认 0 = 全部 280 个）。

## 结果

### GEMV（`gemv_bench.csv`，15 次重复中位数）

![gemv](artifacts/figures/gemv_bench.png)

Llama-7B 形状 4096×4096，权重字节按格式计（fp32 67 MB、fp16 33.5 MB、Q8_0 17.8 MB、Q4_0 9.4 MB）：

| 方法 | 1 线程 µs | GB/s | 4 线程 µs | GB/s |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 4763 | 14.1 | 1156 | 58.1 |
| fp16 | 2312 | 14.5 | 511 | 65.6 |
| Q8_0 浮点融合 | 1494 | 11.9 | 364 | 48.9 |
| Q4_0 先反量化 | 4541 | 2.1 | 1214 | 7.8 |
| Q4_0 浮点融合 | 1743 | 5.4 | 489 | 19.3 |
| Q8_0 × Q8 VNNI | 1009 | 17.7 | 275 | 64.7 |
| Q4_0 × Q8 VNNI | 1034 | 9.1 | 311 | 30.3 |
| Q8_0 × Q8 VNNI，4 累加器 | 903 | 19.8 | 297 | 60.0 |
| Q4_0 × Q8 VNNI，4 累加器 | 990 | 9.5 | 303 | 31.1 |

其他三个形状（GPT-2 MLP 3072×768、Llama-1B 2048² 与 8192×2048）见 `summary.json`；小矩阵全部在缓存里，GB/s 可以到 115。
4 线程加速比：fp32 4.12、Q8 × Q8 3.66、Q4 × Q8 3.32；Q4 的扩展最差，因为它的瓶颈是每核的指令吞吐而不是共享的带宽。

两次空闲运行之间的波动：上一次 bench 里 fp32 4447、Q8 × Q8 1235、Q4 × Q8 1005 µs，这一次 4763、1009、1034。Q8 差 18%，Q4 只差 3%：带宽和缓存状态敏感的内核会漂，指令受限的内核不动。表里只用这一次运行，所有方法在同一次运行里计时。

### 解码上界（`decode_bound.csv`）

tok/s ≤ 带宽 / 权重字节，只算 Linear 权重，不含 KV cache 与激活：

| 模型 | 格式 | 权重 GB | Pi 5（17.1 GB/s） | Orin Nano（68 GB/s） | 本机 4 线程（按实测 GB/s） |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-7B | fp16 | 13.48 | 1.3 | 5.0 | 4.9 |
| Llama-7B | Q8_0 | 7.16 | 2.4 | 9.5 | 9.0 |
| Llama-7B | Q4_0 | 3.79 | 4.5 | 17.9 | 8.0 |
| Llama-1B | Q4_0 | 0.70 | 24.5 | 97.5 | 43.4 |
| GPT-2 124M | Q4_0 | 0.070 | 244.7 | 974.9 | 434.5 |

最后一列用的是 4096² 矩阵上测到的 GB/s，其中 fp16 和 Q8 含 L3 命中，对 GB 级的整模型偏乐观；按 DRAM 流式 48.4 GB/s 封顶，7B Q8_0 是 6.8 tok/s，低于 Q4_0 的 8.0。

### GPT-2 困惑度（`gpt2_ppl.csv`、`gpt2_layer_sqnr.csv`）

wikitext-2 raw test，空行连接，280 个不重叠的 1024 token 窗口（286,440 个预测 token）；量化全部 48 个 Linear（c_attn、c_proj、c_fc、mlp.c_proj），lm_head 与 wte 共享、保持 fp32。

| 格式 | bit/权重 | 困惑度 | 相对 fp32 | 层平均 SQNR dB |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 32 | 29.951 | — | — |
| Q8_0 | 8.5 | 29.953 | +0.007% | 45.08 |
| Q4_1 | 5.0 | 32.15 | +7.3% | 21.96 |
| Q4_0 | 4.5 | 31.88 | +6.4% | 21.03 |

fp32 的 29.95 与 mlsys-infra 的 fp16 29.96 一致，说明两边的前向与评测协议对齐。Q8_0 与 Q4_0 的 SQNR 差 24.04 dB，接近每 4 bit 6.02 × 4 = 24.1 dB。

## 和原课程的差异

- llama.cpp 的 x86 内核还有 AVX2 / AVX-512 多块并行、K-quants（Q4_K 等超块格式）、imatrix；这里只实现最基础的 Q4_0 / Q8_0 / Q4_1，目的是把每一项开销拆开测。
- GPT-2 前向只保留困惑度需要的部分（因果注意力、tanh GELU、pre-LN），不做 KV cache 和采样。

## 这些数字不能推出什么

- GEMV 测的是单层、重复调用；4096² 的矩阵会部分留在 L3，不代表整模型每 token 都从 DRAM 读的情况。
- 解码上界只算权重字节，不含 KV cache、lm_head（GPT-2 里 38.6M 参数）、调度和采样开销。
- Q4_0 与 Q4_1 的困惑度差 0.27，没有做置信区间；「最大幅值元素精确」能否解释它只是假设。
- Pi 5 与 Orin Nano 两列是用设备表带宽算出的上界，不是实测。

## 文件

| 路径 | 内容 |
| --- | --- |
| `code/blockq/pack.py` | Q8_0 / Q4_0 / Q4_1 打包、反量化、SQNR |
| `code/blockq/gpt2.py` | 从 `model.safetensors` 直接构建 GPT-2，`Conv1D` 布局与 HF 一致 |
| `cpp/gemv.cpp` | 打包、9 种 GEMV、`--mode=bench / check / pack` |
| `scripts/run_experiments.py` | `bench`、`gpt2`、`decode_bound`、`summarize` |
| `tests/test_blockq.py` | 块大小、Q4_0 最大元素精确、与 gguf 逐字节一致、C++ 打包一致、内核误差、4 累加器与原版一致、GPT-2 因果性 |
