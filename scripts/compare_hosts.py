"""Cross-host table: the same lab code on the 4-core Ice Lake VM and on a Xeon 8470Q server.

Usage: python scripts/compare_hosts.py
Reads each lab's ``artifacts/raw/summary.json`` (local VM) and
``artifacts/raw/xeon8470q/summary.json`` (server) and rewrites interview/CROSS-HOST.md.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solutions"
OUT = ROOT / "interview" / "CROSS-HOST.md"

# (lab, dotted key into summary.json, label, unit)
METRICS = [
    ("edge-runtime/host-roofline", "t1.dram_read_gbps", "DRAM 读带宽，1 线程", "GB/s"),
    ("edge-runtime/host-roofline", "t4.dram_read_gbps", "DRAM 读带宽，4 线程", "GB/s"),
    ("edge-runtime/host-roofline", "t16.dram_read_gbps", "DRAM 读带宽，16 线程", "GB/s"),
    ("edge-runtime/host-roofline", "t1.l1_read_gbps", "L1 读带宽，1 线程", "GB/s"),
    ("edge-runtime/host-roofline", "t1.fp32_gflops", "fp32 FMA 峰值，1 线程", "GFLOP/s"),
    ("edge-runtime/host-roofline", "t4.fp32_gflops", "fp32 FMA 峰值，4 线程", "GFLOP/s"),
    ("edge-runtime/host-roofline", "t16.fp32_gflops", "fp32 FMA 峰值，16 线程", "GFLOP/s"),
    ("edge-runtime/host-roofline", "t1.int8_gops", "int8 VNNI 峰值，1 线程", "GOP/s"),
    ("edge-runtime/host-roofline", "t16.int8_gops", "int8 VNNI 峰值，16 线程", "GOP/s"),
    ("edge-runtime/host-roofline", "t1.fp32_ridge", "fp32 拐点（DRAM，1 线程）", "FLOP/B"),
    ("edge-runtime/int8-kernels", "gops_square_512_fp32_avx512", "512³ fp32 AVX-512", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_square_512_int8_scalar_autovec", "512³ int8 标量（靠编译器自动向量化）", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_square_512_int8_vnni_packed", "512³ int8 VNNI + 打包", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_square_512_int8_amx_tile", "512³ int8 AMX tile", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_llm_prefill_256x4096x4096_int8_vnni_packed", "prefill 256×4096×4096 VNNI", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_llm_prefill_256x4096x4096_int8_amx_tile", "prefill 256×4096×4096 AMX", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_mbv2_pw_expand_56_int8_vnni_packed", "MBv2 expand 56（K=24）VNNI", "GOP/s"),
    ("edge-runtime/int8-kernels", "gops_mbv2_pw_expand_56_int8_amx_tile", "MBv2 expand 56（K=24）AMX", "GOP/s"),
    ("edge-runtime/conv-kernels", "gflops_resnet_3x3_56_implicit_gemm", "3×3 conv 56² implicit GEMM", "GFLOP/s"),
    ("edge-runtime/conv-kernels", "gflops_resnet_3x3_56_winograd_f2x3", "3×3 conv 56² Winograd F(2,3)", "GFLOP/s"),
    ("edge-runtime/conv-kernels", "gflops_resnet_3x3_14_implicit_gemm", "3×3 conv 14² implicit GEMM", "GFLOP/s"),
    ("edge-runtime/conv-kernels", "gflops_resnet_3x3_14_winograd_f2x3", "3×3 conv 14² Winograd F(2,3)", "GFLOP/s"),
    ("edge-runtime/conv-kernels", "gflops_mbv2_dw_56_dw_nhwc_simd", "depthwise 56² NHWC SIMD", "GFLOP/s"),
    ("edge-runtime/dma-double-buffer", "tile64k_work4:speedup", "双缓冲实测加速（64 KiB tile × 512，work=4）", "×"),
    ("edge-runtime/dma-double-buffer", "tile64k_work4:model_speedup", "双缓冲模型预测（同上）", "×"),
    ("edge-runtime/dma-double-buffer", "tile1024k_work4:speedup", "双缓冲实测加速（1 MiB tile，work=4）", "×"),
    ("mlc/tensor-program-tuning", "square_256.overall_best_gflops", "256³ 调度搜索最优", "GFLOP/s"),
    ("mlc/tensor-program-tuning", "square_256.numpy_1thread_gflops", "256³ NumPy 单线程（OpenBLAS）", "GFLOP/s"),
    ("mlc/tensor-program-tuning", "square_256.cost_model_holdout_spearman", "代价模型留出集 Spearman", ""),
    ("edge-runtime/q4-gemv", "llama7b_4096x4096:fp32:t1:GBps", "GEMV 4096² fp32，1 线程", "GB/s"),
    ("edge-runtime/q4-gemv", "llama7b_4096x4096:q4_0_x_q8_vnni:t1:us", "GEMV 4096² Q4_0×Q8 VNNI，1 线程", "µs"),
    ("edge-runtime/q4-gemv", "llama7b_4096x4096:q4_0_x_q8_vnni:t4:us", "GEMV 4096² Q4_0×Q8 VNNI，4 线程", "µs"),
    ("edge-runtime/q4-gemv", "llama7b_4096x4096:q4_0_x_q8_vnni:t16:us", "GEMV 4096² Q4_0×Q8 VNNI，16 线程", "µs"),
    ("edge-runtime/q4-gemv", "llama7b_mlp_11008x4096:fp32:t16:GBps", "GEMV 11008×4096 fp32，16 线程", "GB/s"),
]


def lookup(d: dict, key: str):
    if key in d:
        return d[key]
    cur = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def load(lab: str, host: str | None) -> dict:
    raw = SOL / lab / "artifacts" / "raw"
    p = raw / host / "summary.json" if host else raw / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def fmt(v) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def main() -> None:
    cache: dict[tuple[str, str | None], dict] = {}
    rows = []
    for lab, key, label, unit in METRICS:
        for host in (None, "xeon8470q"):
            cache.setdefault((lab, host), load(lab, host))
        a, b = lookup(cache[(lab, None)], key), lookup(cache[(lab, "xeon8470q")], key)
        ratio = f"{b / a:.2f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else "—"
        rows.append(f"| {lab.split('/')[-1]} | {label} | {unit} | {fmt(a)} | {fmt(b)} | {ratio} |")
    text = "\n".join([
        "# 同一份代码，两台主机",
        "",
        "> 由 `python scripts/compare_hosts.py` 从各 lab 的 `artifacts/raw/summary.json`（本地）与",
        "> `artifacts/raw/xeon8470q/summary.json`（服务器）生成，不要手改。",
        "",
        "| 主机 | CPU | 核 / 线程 | ISA | 编译器 | 说明 |",
        "|---|---|---|---|---|---|",
        "| 本地 VM | Xeon Ice Lake-SP（Family 6 Model 106） | 4 / 8 | AVX-512 + VNNI | MinGW GCC 13.2 | Windows Server 2022 |",
        "| 服务器 | Xeon Platinum 8470Q（Sapphire Rapids）×2 | 104 / 208（cgroup 限 22 核） | AVX-512 + VNNI + **AMX-INT8** "
        "| GCC 11.4 | Ubuntu 22.04，绑定 NUMA1 的 16 个物理核（60–75） |",
        "",
        "比值 = 服务器 / 本地；对 µs 类指标，比值 < 1 表示服务器更快。",
        "",
        "![两台主机的加速比（µs 指标已取倒数，>1 均表示服务器更快）](figures/cross_host.png)",
        "",
        "| lab | 指标 | 单位 | 本地 VM | Xeon 8470Q | 比值 |",
        "|---|---|---|---:|---:|---:|",
        *rows,
        "",
        "## 读表要点",
        "",
        "1. **单核差距只有 1.3–1.6 倍，多核才拉开。** 1 线程 DRAM 读带宽 13.4 → 17.5 GB/s，"
        "单核受限于每核在途缺失数（line fill buffer），"
        "而不是内存控制器；16 线程到 193 GB/s，是本地 4 线程（48 GB/s）的 4 倍。面试里“服务器内存带宽几百 GB/s”说的是全插槽，"
        "单线程 GEMV 只能拿到其中不到 10%。",
        "2. **AMX 在大 K 上是 VNNI 的 3–6 倍，在小 K 上反而更慢。** 512³ 上 1704 对 563 GOP/s，prefill 形状上 1659 对 280；"
        "MBv2 expand（K=24）被补齐到 tile 的 K=64，有效算力只剩 3/8，162 对 270。选 kernel 要看 K 能否填满 tile。",
        "3. **“标量代码”的速度取决于编译器。** 同一份标量 int8 循环，本地 GCC 13 下 100 GOP/s、服务器 GCC 11 下 20 GOP/s——"
        "对 `int8.cpp` 用 `-S -march=native` 编出汇编计数：GCC 13 全文件 `vpdpbusd` 78 条、字节扩展 `vpmov[sz]xbw` 14 条；"
        "GCC 11 是 60 条与 80 条——多出的 18 条 VNNI 就是标量循环被 GCC 13 识别成点积，GCC 11 只能先扩成 16 位再乘。"
        "手写 intrinsics 的版本两边都稳定，这就是手写的理由。",
        "4. **Winograd 的收益依赖主机。** 本地在 56² 上 Winograd 比 implicit GEMM 快 16%，服务器上反而慢 32%（54 对 80）；"
        "14² 上本地快 2.1 倍、服务器快 3.0 倍。56² 时变换的额外访存在更快的 GEMM 面前显得更贵。",
        "5. **软件双缓冲在服务器上全部变慢。** 64 KiB × 512 块、work=4 时本地 1.08 倍、服务器 0.70 倍，"
        "而模型预测 1.6 倍：拷贝线程与计算线程"
        "共享同一插槽的带宽与 cgroup 配额，并不是独立 DMA 引擎。"
        "模型只在拷贝真正不占计算资源时成立（MCU 的 DMA、GPU 的 copy engine）。",
        "6. **搜索收敛更快，但代价模型排序能力没有本质提升。** 两边 Spearman 都在 0.47–0.56；"
        "服务器上 random 在 13 次试验内到 90%，"
        "本地要 20 次；一种解释是 Sapphire Rapids 的 L2 更大（2 MB 对 1.25 MB），好调度的区域更宽，但本实验没有单独验证。",
        "",
    ])
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
    plot(cache)


PLOT_LABEL = {
    "t1.dram_read_gbps": "DRAM read, 1 thr",
    "t4.dram_read_gbps": "DRAM read, 4 thr",
    "t1.l1_read_gbps": "L1 read, 1 thr",
    "t1.fp32_gflops": "fp32 FMA peak, 1 thr",
    "t4.fp32_gflops": "fp32 FMA peak, 4 thr",
    "t1.int8_gops": "int8 VNNI peak, 1 thr",
    "gops_square_512_fp32_avx512": "512³ fp32 AVX-512",
    "gops_square_512_int8_scalar_autovec": "512³ int8 scalar (auto-vec)",
    "gops_square_512_int8_vnni_packed": "512³ int8 VNNI packed",
    "gops_mbv2_pw_expand_56_int8_vnni_packed": "MBv2 K=24 VNNI packed",
    "gflops_resnet_3x3_56_implicit_gemm": "conv 3×3 56² implicit GEMM",
    "gflops_resnet_3x3_56_winograd_f2x3": "conv 3×3 56² Winograd",
    "gflops_resnet_3x3_14_implicit_gemm": "conv 3×3 14² implicit GEMM",
    "gflops_resnet_3x3_14_winograd_f2x3": "conv 3×3 14² Winograd",
    "gflops_mbv2_dw_56_dw_nhwc_simd": "depthwise 56² NHWC",
    "tile64k_work4:speedup": "double-buffer speedup (64K, w=4)",
    "square_256.overall_best_gflops": "tuned 256³ best",
    "square_256.numpy_1thread_gflops": "NumPy 256³ 1 thr",
    "llama7b_4096x4096:fp32:t1:GBps": "GEMV 4096² fp32, 1 thr",
    "llama7b_4096x4096:q4_0_x_q8_vnni:t1:us": "GEMV 4096² Q4×Q8, 1 thr",
    "llama7b_4096x4096:q4_0_x_q8_vnni:t4:us": "GEMV 4096² Q4×Q8, 4 thr",
}


def plot(cache: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = list(dict.fromkeys(lab for lab, *_ in METRICS))
    colors = dict(zip(labs, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    items = []
    for lab, key, _label, unit in METRICS:
        if key not in PLOT_LABEL:
            continue
        a, b = lookup(cache[(lab, None)], key), lookup(cache[(lab, "xeon8470q")], key)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and a and b):
            continue
        items.append((PLOT_LABEL[key], a / b if unit == "µs" else b / a, lab))
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(items) + 1.2))
    ys = range(len(items))[::-1]
    ax.barh(list(ys), [v - 1 for _, v, _ in items], left=1, color=[colors[lab] for *_, lab in items])
    for y, (_, v, _) in zip(ys, items):
        ax.text(v * (1.04 if v >= 1 else 0.96), y, f"{v:.2f}×", va="center", ha="left" if v >= 1 else "right",
                fontsize=7)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.12, 4.5)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4], ["1/8", "1/4", "1/2", "1", "2", "4"])
    ax.set_yticks(list(ys), [n for n, _, _ in items], fontsize=8)
    ax.set_xlabel("Xeon 8470Q relative to the 4-core Ice Lake VM (>1 = server faster)")
    ax.set_title("Same code, two hosts: most kernels gain 1.3–2.7×, two lose", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[lab]) for lab in labs]
    ax.legend(handles, [lab.split("/")[-1] for lab in labs], fontsize=7, loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    out = ROOT / "interview" / "figures" / "cross_host.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
