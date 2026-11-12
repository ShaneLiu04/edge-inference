# 仓库完整性检查

这份文件记录「怎么知道仓库里的东西是对的」：跑了哪些检查，结果如何，数字的适用范围在哪里，还有哪些已知的缺口。

最后检查：2026-11-12。

## 自动检查

| 检查什么 | 命令 | 结果 |
| --- | --- | --- |
| 笔记的结构（16 节、元数据、公式块、追问链）、到 mlsys-infra 笔记的链接、索引、术语 | `python notes/check_notes.py` | 8 讲全部通过 |
| 全部测试，AVX-512 + VNNI | `./scripts/run-all-tests.ps1` | 11 套全部通过，140 个测试 |
| 全部测试，无 AVX-512 的回退路径 | `$env:EDGE_MARCH = "haswell"; ./scripts/run-all-tests.ps1` | 上一轮 10 套 135 个。runtime-compare 是纯 Python，本轮没有重跑这条 |
| README、笔记、报告里的关键数字与原始数据一致 | `python scripts/check_claims.py`（对照 `claims.json`） | 123 条全部吻合 |
| Markdown 相对链接、实验目录结构、报告（PDF 存在、图能找到、每条参考文献都被引用且没有未定义的引用）、claims、密钥扫描 | `./scripts/check-repo.ps1`（报告部分由 `scripts/check_reports.py` 完成） | 通过，10 个实验，11 份报告 |
| 代码风格 | `python -m ruff check .` | 通过 |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：Ubuntu + Python 3.12，CPU torch，ruff + `check-repo.ps1` + `run-all-tests.sh`；没有 mlsys-infra 同级目录时跳过跨仓库笔记名校验 | 配置就绪。Gitee 不原生运行 GitHub Actions，镜像到 GitHub 后生效 |

## 每套测试检查了什么

| 套件 | 测试数 | 最关键的正确性检查 |
| --- | ---: | --- |
| tools | 15 | 公式计算器的闭式值；first-fit 碎片反例；int32 累加上界；claims 的 `expect` / `ratio` / `equals` 语义与失败信息；编译缓存；自旋线程池每个下标只执行一次 |
| host-roofline | 6 | 带宽平台识别不被噪声切出新平台；拐点；C++ 微基准编译运行 |
| int8-kernels | 33 | int32 累加 C++ 与 Python 参考逐位相等（6 种层）；requant 与 ORT `QLinearConv` 逐位一致；构造平局区分 half-even 与 half-away；maddubs 饱和边界；8 种 GEMM 与参考逐位相等 |
| conv-kernels | 14 | 7 种卷积实现对 fp64 参考；Winograd 的 Toom-Cook 构造复现 Lavin 的 F(2,3) 矩阵，1D / 2D / r = 5 在 fp64 下精确；fp32 误差随 tile 增大 |
| dma-double-buffer | 11 | 事件模拟与 E15、E16 闭式在随机参数上相等；共享 DMA 通道；分块搜索满足 L1 约束（E17） |
| q4-gemv | 10 | Q8_0 / Q4_0 / Q4_1 的 NumPy 与 C++ 打包都与 gguf 参考逐字节一致；Q4_0 最大幅值元素精确；9 种 GEMV 对 fp32 的误差，4 累加器版与 1 累加器版差 < 1e-6；GPT-2 前向是因果的 |
| llama-cpp-7b | 5 | 用真实输出片段测解析器：llama-bench jsonl、llama-perplexity 的最终 PPL 与 KLD 统计、llama-quantize 的大小；不需要 llama.cpp |
| lab2-quantization | 16 | BN 折叠前后输出一致；逐张量把小通道量化成 0；校准器在重尾分布上的行为；零尖峰把 KL 阈值拉低；STE 梯度在裁剪区外为 0；16 bit 伪量化接近浮点 |
| tinyengine-memory | 9 | 规划结果互不重叠且不低于下界；first-fit 碎片而贪心不碎；trace 融合 BN / ReLU 并保持残差输入存活；原地 depthwise 降低峰值；halo 回推 |
| tensor-program-tuning | 16 | 编译出的调度与参考 GEMM 一致；分块必须整除；策略不超预算、空间比种群小时能终止；岭回归恢复线性函数 |

q4-gemv 的 Triton 内核不在 pytest 里：`python gpu/bench_gpu.py check` 在 RTX 6000D 上对照 fp32 参考（Q8_0 / Q4_0 GEMV 相对误差 < 2e-3，Q4_0 Tensor Core GEMM < 5e-3），通过后才计时。

## 数字的适用范围

读任何一个数字之前，先确认它是在哪台机器上、用什么方法得到的：

- **本地虚拟机**：Windows Server，Ice Lake 服务器 CPU，4 物理核 / 8 逻辑核，AVX-512F + VNNI；MinGW g++ 13.2，Python 3.12，numpy 2.x，torch 2.6 CPU，ONNX Runtime 1.31。
  一个后台系统服务会间歇占用约一个核（总负载 16%–18%），计时前确认负载在 5% 左右；同一时间只跑一个计时任务。时钟按 fp32 峰值反推约 2.42 GHz（系统报告 2.8 GHz）。
- **服务器 CPU**：AutoDL 上的 2 × Xeon Platinum 8470Q（Sapphire Rapids，AVX-512 + VNNI + AMX-INT8），GCC 11.4，Ubuntu 22.04。
  计时绑 NUMA1 的 16 个物理核（60–75）；容器 cgroup 只给 22 个 CPU 的配额，宿主上有其他租户，被限流的点（标准差与均值同量级）丢掉重跑。原始数据在各实验的 `artifacts/raw/xeon8470q/`。
- **服务器 GPU**：RTX 6000D（sm_120，84 GB，实测读 1333 GB/s），CUDA 12.8，Triton；计时用 CUDA event，「L2 冷」指轮换总量 ≥ 4 GB 的权重拷贝。原始数据在 `q4-gemv/artifacts/raw/gpu/` 和 `llama-cpp-7b/artifacts/raw/`。
- **跨主机比值**混合了硬件与编译器（GCC 13 对 GCC 11）；[CROSS-HOST.md](interview/CROSS-HOST.md) 里用汇编计数把编译器因素单独拆出来。
- **性能计数器**：两台机器都没有（虚拟机没有 PMU，容器里没有 perf 权限，`ncu` 报 `ERR_NVGPUCTRPERM`）。端口压力、缓存命中、降频、K-quant 的指令受限都是推算，README、笔记和报告里已标注。
- **MCU、NPU、Pi 5、Orin** 的数字全部来自 `edge_harness/devices.py` 的数据手册参数或模型（事件模拟、Dory 式分块、带宽上界），不是实测。
- **7B 模型**：q4-gemv 的 token/s 是只算 196 个 Linear 的预测；完整模型的端到端数字来自 llama-cpp-7b（llama.cpp `149f564`），两者的差在该实验 README 里拆过。
- **精度结论**：lab2 的单次表格来自 CPU 训练的 seed 0；结论是否成立看五个种子的配对差与 95% 置信区间（t 分布，n = 5）。

## 已知缺口

- 没有开发板：没有在 MCU、NPU、Pi 上跑过任何一个实验。
- [runtime-compare](solutions/edge-runtime/runtime-compare/README.md) 在 x86 服务器 CPU 上对照了 ORT、LiteRT、ExecuTorch、TVM；TFLite Micro 和 MetaSchedule 调优过的 TVM 没有跑。本仓库的搜索器、规划器、块格式是最小实现，与官方实现对齐的部分（gguf 逐字节、ORT 逐位）在各 README 写明。
- runtime-compare 的时间是 0.5 核 cgroup 下的 CPU 时间：同一批内部的排序可靠，跨批的绝对值能差一半。
- 官方 starter code 未镜像，见 [UPSTREAM.md](UPSTREAM.md)。
- llama.cpp 端到端比 q4-gemv 预测多出的 0.41–0.47 ms/token 没有用 nsys 拆到具体算子；llama-perplexity 的 ctx 512 协议与 mlsys-infra 的 HF 协议不同，两边困惑度不能直接比。
- 报告只做静态检查（`check_reports.py` 不需要 TeX）；重新编译需要 XeLaTeX：`latexmk -xelatex <report>.tex`。

## 自己复验

```powershell
$env:PYTHONPATH = (Get-Location).Path
python notes/check_notes.py
./scripts/run-all-tests.ps1
./scripts/check-repo.ps1
python -m ruff check .
```
