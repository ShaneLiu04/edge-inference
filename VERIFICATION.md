# 仓库完整性检查

最后检查：2026-11-03。

## 自动检查

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 笔记结构（16 节、元数据、公式块、追问链）、到 mlsys-infra 笔记的链接、索引、术语 | `python notes/check_notes.py` | 8 讲通过 |
| 全部测试，AVX-512 + VNNI | `./scripts/run-all-tests.ps1` | 10 套全部 PASS，135 个测试 |
| 全部测试，无 AVX-512 的回退路径 | `$env:EDGE_MARCH = "haswell"; ./scripts/run-all-tests.ps1` | 10 套全部 PASS，135 个测试 |
| README 与笔记里的关键数字对照 `artifacts/raw/` | `python scripts/check_claims.py` | 90 条全部吻合 |
| Markdown 相对链接、实验目录结构、claims、密钥扫描 | `./scripts/check-repo.ps1` | 通过，8 个实验 |
| 代码风格 | `python -m ruff check .` | 通过 |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：Ubuntu + Python 3.12，CPU torch，ruff + `check-repo.ps1` + `run-all-tests.sh`；没有 mlsys-infra 同级目录时跳过跨仓库笔记名校验 | 配置就绪（Gitee 不原生运行 GitHub Actions，镜像到 GitHub 后生效） |

## 测试清单

| 套件 | 测试数 | 关键的正确性检查 |
| --- | ---: | --- |
| tools | 15 | 公式计算器的闭式值；first-fit 碎片反例；int32 累加上界；claims 的 `expect` / `ratio` / `equals` 语义与失败信息；编译缓存；自旋线程池每个下标只执行一次 |
| host-roofline | 6 | 带宽平台识别不被噪声切出新平台；拐点；C++ 微基准编译运行 |
| int8-kernels | 33 | int32 累加 C++ 与 Python 参考逐位相等（6 种层）；requant 与 ORT `QLinearConv` 逐位一致；构造平局区分 half-even 与 half-away；maddubs 饱和边界；8 种 GEMM 与参考逐位相等 |
| conv-kernels | 14 | 7 种卷积实现对 fp64 参考；Winograd 的 Toom-Cook 构造复现 Lavin 的 F(2,3) 矩阵，1D / 2D / r = 5 在 fp64 下精确；fp32 误差随 tile 增大 |
| dma-double-buffer | 11 | 事件模拟与 E15、E16 闭式在随机参数上相等；共享 DMA 通道；分块搜索满足 L1 约束（E17） |
| q4-gemv | 10 | Q8_0 / Q4_0 / Q4_1 的 NumPy 与 C++ 打包都与 gguf 参考逐字节一致；Q4_0 最大幅值元素精确；9 种 GEMV 对 fp32 的误差，4 累加器版与 1 累加器版差 < 1e-6；GPT-2 前向是因果的 |
| lab2-quantization | 16 | BN 折叠前后输出一致；逐张量把小通道量化成 0；校准器在重尾分布上的行为；零尖峰把 KL 阈值拉低；STE 梯度在裁剪区外为 0；16 bit 伪量化接近浮点 |
| tinyengine-memory | 9 | 规划结果互不重叠且不低于下界；first-fit 碎片而贪心不碎；trace 融合 BN / ReLU 并保持残差输入存活；原地 depthwise 降低峰值；halo 回推 |
| tensor-program-tuning | 16 | 编译出的调度与参考 GEMM 一致；分块必须整除；策略不超预算、空间比种群小时能终止；岭回归恢复线性函数 |

## 数字的边界

- 实验机器：Windows Server 虚拟机，Ice Lake 服务器 CPU，4 物理核 / 8 逻辑核，AVX-512F + VNNI；MinGW g++ 13.2，Python 3.12，numpy 2.x，torch 2.6 CPU，ONNX Runtime 1.31。
- 计时只在空闲机器上做：一个后台系统服务会间歇占用约一个核（总负载 16%–18%），计时前确认负载在 5% 左右；同一时间只跑一个计时任务。
- 虚拟机没有性能计数器：端口压力、缓存命中、降频都是推算，README 与笔记里已标注。时钟按 fp32 峰值反推约 2.42 GHz（系统报告 2.8 GHz）。
- MCU、NPU、Pi 5、Orin 的数字全部来自 `edge_harness/devices.py` 的数据手册参数或模型（事件模拟、Dory 式分块、带宽上界），不是实测。
- 7B 模型的 token/s 由单层 GEMV 推算；没有跑完整 7B 模型。

## 已知缺失

- 没有开发板：没有在 MCU、NPU、Pi 上跑过任何一个实验。
- 没有 TVM、ExecuTorch、TFLite Micro 的端到端对照；本仓库的搜索器、规划器、块格式是最小实现，与官方实现对齐的部分（gguf 逐字节、ORT 逐位）在各 README 写明。
- 官方 starter code 未镜像，见 [UPSTREAM.md](UPSTREAM.md)。
- 一次训练、一个种子的精度结论（lab2）：88% 附近 0.1 个点的差在单次评测的标准误（约 0.3 个点）以内。

## 复验

```powershell
$env:PYTHONPATH = (Get-Location).Path
python notes/check_notes.py
./scripts/run-all-tests.ps1
./scripts/check-repo.ps1
python -m ruff check .
```
