# 面试准备

| 文件 | 什么时候打开 |
| --- | --- |
| [HARDWARE-NUMBERS.md](HARDWARE-NUMBERS.md) | 所有估算的起点：十个数、本机实测屋顶、设备表，以及哪些数只是上界 |
| [FORMULAS.md](FORMULAS.md) | 对方说「写一下」「估一下」。E1–E27，编号与笔记一致，每条带口算、实测锚点和常见错答 |
| [SYSTEM-DESIGN.md](SYSTEM-DESIGN.md) | 设计题：256 KB MCU 视觉唤醒词、手机 1B LLM、NPU 算子回退、量化验收流水线 |
| [PORTFOLIO.md](PORTFOLIO.md) | 讲自己做过的实验。6 个故事，每条有 20 秒结论、白板和「不能推出的结论」 |
| [CODING.md](CODING.md) | 手写题 C1–C22：requant、BN 折叠、KL 校准、arena 规划、Winograd、VNNI 打包、Q4 点积 |
| [QUESTION-BANK.md](QUESTION-BANK.md) | 42 道按模块分组的概念题，每题带常见错答；M09 是换机器后翻转的结论 |
| [CROSS-HOST.md](CROSS-HOST.md) | 同一代码在 4 核 VM 与 Xeon 8470Q 上的 34 项指标对照，附加速比图 |
| [COURSE-QA.md](COURSE-QA.md) | 8 讲的一句话答案 |
| [ENGINEERING.md](ENGINEERING.md) | 「掉点 / 对不上 / 变慢 / 放不下怎么查」的排查顺序 |

数据中心一侧（GPU、分布式训练、LLM serving）的同名文件在 [mlsys-infra/interview](https://gitee.com/liu-xingyan04/mlsys-infra/tree/main/interview)，两边的对应关系见 [BRIDGE.md](../BRIDGE.md)。

## 三种面试怎么准备

**概念面（30–45 分钟）**：先过 [COURSE-QA](COURSE-QA.md)，再按岗位挑题库的两个模块。
每题先说结论，再说本仓库测到的数，最后主动说一个常见错答。

**估算 / 设计面（45–60 分钟）**：背 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md) 的十个数，按 [FORMULAS](FORMULAS.md) 末尾的岗位顺序默写五条，
然后限时 20 分钟做一道 [设计题](SYSTEM-DESIGN.md)。先报估算，再画框图，最后讲取舍。

**手写代码面（45–60 分钟）**：[CODING](CODING.md) 的每道题都有参考实现和测试。关掉参考实现写一遍，跑对应测试，再对比。

## 讲实验的规则

本仓库的实验先在一台 4 物理核的 Ice Lake 虚拟机上跑，再在 AutoDL 的 Xeon 8470Q + RTX 6000D 服务器上复现（结果放在各 lab 的
`artifacts/raw/xeon8470q/` 与 `artifacts/raw/gpu/`）。没有开发板，两台机器都没有性能计数器。能讲的是：

- **正确性**：与参考实现逐位或逐字节相等（整数卷积、requant、块量化打包）。
- **同机相对比较**：A 比 B 快几倍、离实测屋顶多远、规划器离下界多远。
- **模型输出**：双缓冲事件模拟、Dory 式分块、由设备表带宽推出的 token/s 上界。

不能讲成「我在 MCU / NPU / Pi 上测到了多少」。每个故事的「不能推出的结论」就是为这个准备的。
