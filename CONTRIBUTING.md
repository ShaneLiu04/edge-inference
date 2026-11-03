# 贡献约定

和 [mlsys-infra](https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/CONTRIBUTING.md) 同一套规则。改动合入前本地跑两条命令，CI 跑同样的两条：

```powershell
./scripts/check-repo.ps1      # 笔记结构、Markdown 链接、lab 布局、claims、密钥
./scripts/run-all-tests.ps1   # 9 个测试套件，各自一个 pytest 进程
```

Linux 上用 `PYTHON=python3 ./scripts/run-all-tests.sh`。想模拟只有 AVX2 的机器（CI 的 runner 通常如此），设 `EDGE_MARCH=haswell` 再跑测试，所有 C++ 会按该 ISA 重新编译，AVX-512/VNNI 专属的方法自动不参与比较。

## 一个实验目录必须有

```text
solutions/<course>/<lab>/
  README.md                 结论先行；每个数字能在 artifacts/raw/ 里找到
  code/                     Python 参考实现（可 import，可测试）
  cpp/                      可选；C++ 内核，经 edge_harness.cpp.build 编译
  tests/                    逐位或闭式的正确性测试，不测速度
  scripts/run_experiments.py  --only 选步骤；结果写 artifacts/
  artifacts/raw/            JSONL / CSV / summary.json（入库）
  artifacts/figures/        PNG（入库）
```

`cpp/bin/`、`artifacts/work/`、`checkpoints/`、`.data/` 不入库。

## 数字

- 计时报 p50，并在原始文件里保留 p10、p90；C++ 侧用 `edgebench.h` 的 `time_fn`，Python 侧用 `edge_harness.timing`。
- 计时任务串行跑，不和训练、其他计时同时跑；线程数写进结果行。
- README 里引用的数在 `claims.json` 里登记，`python scripts/check_claims.py` 校验；改了实验就重跑并更新 claims，而不是改 README 里的数。
- 精度类结果报多个种子时写均值和标准差。

## 笔记

- 模板在 [templates/lecture-note.md](templates/lecture-note.md)，校验器 `python notes/check_notes.py --verbose`。
- 每篇在「关联笔记」里至少链一篇 mlsys-infra 的对应模块（Gitee 链接）。
- 公式编号用 E 前缀（E1、E2 …），在 [interview/FORMULAS.md](interview/FORMULAS.md) 里有同号条目。

## 代码

- Python 用 ruff（`python -m ruff check .`，规则见 `pyproject.toml`）。
- C++17，`-O3 -march=native`；ISA 专属代码放在 `#if defined(__AVX512VNNI__)` 一类守卫里，并提供可移植回退。
- Windows/MinGW 下 AVX-512 栈只按 16 字节对齐，`edge_harness.cpp` 自动加 `-Wa,-muse-unaligned-vector-move`；不要在 lambda 里按值捕获 `__m512i`。

## 不入库

令牌、密码、带账号的下载链接；课程官方的 starter 代码与讲义（只放链接，见 [UPSTREAM.md](UPSTREAM.md)）；模型权重和数据集（脚本会下载到 `.data/`）。
