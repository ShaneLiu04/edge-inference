# MIT 6.5940 Lab 3（NAS）资源记录

记录日期：2026-11-12。用来后续整理：哪些资源有公开链接、各自试过什么、结果如何。实验结果不写在这里，见 [lab3-nas](../solutions/mit6.5940/lab3-nas/README.md) 与 [project-mcu-e2e](../solutions/mit6.5940/project-mcu-e2e/README.md)。

## 1. 课程分发的两个压缩包（已镜像到本仓库）

| 文件 | 内容 | 大小（公开笔记记录） | 链接 |
| --- | --- | --- | --- |
| `mcunetv2-dev-main.zip` | OFA-MCUNet 超网代码（`OFAMCUNets`）、`vww_supernet.pth`、精度数据集 `acc_datasets` | 约 31.7 MB | `https://www.dropbox.com/s/3y2n2u3mfxczwcb/mcunetv2-dev-main.zip?dl=0` |
| `data.zip` | VWW 验证集 `vww-s256`（`data/vww-s256/val`） | 约 376 MB | `https://www.dropbox.com/s/169okcuuv64d4nn/data.zip?dl=0` |

链接来源：TinyML KOR 的 Lab 3 笔记（https://tinyml-kor.github.io/blog/posts/labs/lab03.html）与 quidproquo 的 Lab 3 长文（https://quidproquo.cc/posts/ai/2026-09-30-mit-65940-lab3-nas-en/，写于 2026-10-01 时两个链接仍可用）。

2026-11-19 直连 Dropbox 下载成功，两个包已镜像到本仓库 `assignments/lab3-resources/`。Gitee 社区版仓库不支持 Git LFS、单文件上限 100 MB，所以 `data.zip` 按 90 MiB 切成 4 个 `.part` 文件（`part00`–`part02` 各 94,371,840 B，`part03` 92,844,928 B），克隆后合并即可还原。校验值（SHA256）：

| 文件 | 大小 | SHA256 |
| --- | ---: | --- |
| `mcunetv2-dev-main.zip` | 31,675,052 B | `8946f8093a9073f513a58b36a108209b3d49e0b391e71a73506bd1de897463ec` |
| `data.zip` | 375,960,448 B | `623782584192360042be870381a4ab5a161c4b96a8005e1af8f75eae6507d5dc` |

解压方法：`tar -xf assignments/lab3-resources/mcunetv2-dev-main.zip`（得到 `mcunetv2-dev-main/lab3.ipynb`、`vww_supernet.pth`、`acc_datasets/`）。`data.zip` 先合并再解压：Linux/macOS 用 `cat assignments/lab3-resources/data.zip.part* > data.zip`，Windows 用 `cmd /c copy /b assignments\lab3-resources\data.zip.part00+data.zip.part01+data.zip.part02+data.zip.part03 data.zip`；然后 `tar -xf data.zip` 得到 `data/vww-s256/val`。

下载结果：
- 2026-11-12 本机通过公司代理：Dropbox 返回 403（`CONNECT tunnel failed`）；本机直连：连接超时（curl 退出码 28）；AutoDL 服务器：连不上 Dropbox。
- 2026-11-19 本机直连 Dropbox 下载成功，内容核对无误后提交到本仓库。

## 2. 官方 MCUNet 仓库（已下载，但不含超网）

- 仓库：https://github.com/mit-han-lab/mcunet（分支 `master`、`dev/persondet`，master 提交 `9c164f8`）。
- README 同时写着 MCUNet（NeurIPS 2020）与 MCUNetV2（NeurIPS 2021）。`https://github.com/mit-han-lab/mcunetv2` 返回 404，MCUNetV2 代码就在这个仓库里。
- 仓库里只有 `mcunet/tinynas/elastic_nn/networks/ofa_proxyless.py`，**没有** `ofa_mcunets.py`（即 `OFAMCUNets`），也没有 `vww_supernet.pth`。
- 本机已下载 master 的源码快照（8.5 MB）到 `D:\STUDY\_dl\mcunet-master.zip`，并解压到 `D:\STUDY\_dl\mcunet-master`。这两个路径在仓库之外，不提交。

## 3. 其他来源（都不能替代课程包）

| 来源 | 结果 |
| --- | --- |
| `https://hanlab18.mit.edu/projects/tinyml/mcunet/release/` | 目录列表 403。固定模型的权重可下（如 `mcunet-320kb-1mb_vww.pth`，2.78 MB），但那是 VWW 固定网络，不是超网。`vww_supernet.pth` 返回 404。 |
| Hugging Face `mit-han-lab/vww-s256` | 数据集仓库是空的（只有 README 与 `.gitattributes`）。 |
| `https://data.brainchip.com/dataset-mirror/vww/vw_coco2014_96_trainval.tar.gz` | 公开可下，224 MB。但它是 MLPerf Tiny 的 96×96 版本，不是 Lab 3 用的 256 边长 `vww-s256`，分辨率对不上。 |
| `https://www.silabs.com/.../vw_coco2014_96.tar.gz` | 公司代理返回 403。 |
| GitHub 搜索 `vww_supernet` / `mcunetv2-dev` | 无结果。 |
| Wayback Machine CDX | 403。 |
| `https://github.com/swagholikar29/Neural-Architecture-Search` | 一位学生的 NAS 作业仓库（`NAS_ntbk.ipynb`，约 4 MB），基于同一个超网思路。未核实内容，也没有权重。 |

## 4. 题面中需要注意的一点

quidproquo 的长文指出：随机搜索单元里约束字典用的键是 `millonMACs`（少了一个 i），而预测器返回的键是 `millionMACs`。`satisfy_constraint` 会跳过约束字典里没有的键，所以两条 MACs 约束可能根本没生效。进化搜索单元拼写正确。做时要先检查这一点。

## 5. 后续要做的事

1. ~~下载上面两个 zip~~ 已完成并随仓库同步：包在 `assignments/lab3-resources/`（`data.zip` 拆成 4 个分段，先合并再解压）。
2. 解压后确认有 `ofa_mcunets.py`、`vww_supernet.pth`、`acc_datasets` 和 `data/vww-s256/val`（本机已确认）。
3. ~~开服务器，装 `thop`、`onnx`，跑 Q1 到 Q10~~ 已完成，结果在 [lab3-nas](../solutions/mit6.5940/lab3-nas/README.md)。上面 `millonMACs` 的拼写问题确认存在：照抄那一格时 Q6 的“MACs ≤ 50M”约束不生效，返回 126.8M MACs 的子网。
4. ~~期末项目（子网 → int8 → 内存规划与 patch → MLPerf Tiny 四列）~~ 已完成，见 [project-mcu-e2e](../solutions/mit6.5940/project-mcu-e2e/README.md)。
