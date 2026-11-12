"""Parse test_linear.log and e2e.log into summary.json and one figure."""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parents[2]))

from edge_harness.writer import write_json  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
IMPS = ["reference", "loop_unrolling", "multithreading", "simd_programming", "multithreading_loop_unrolling",
        "all_techniques"]


def parse_kernels(text: str) -> dict:
    out = {}
    for name in IMPS:
        ok = re.search(rf"Sanity check of {name} implementation: (\w+)!", text)
        row = re.search(rf"^{name}, ([\d.]+), ([\d.]+), (\d+), ([\d.]+)", text, flags=re.M)
        out[name] = {"passed": bool(ok and ok.group(1) == "Passed"), "avg_ms": float(row.group(2)),
                     "gops": float(row.group(4))}
    return out


def parse_e2e(text: str) -> dict:
    out = {}
    for block in re.split(r"^(?==== IMP )", text, flags=re.M):
        head = re.match(r"=== IMP (\d)", block)
        row = re.search(r"^Inference latency, ([\d.]+), ([\d.]+), (\d+)", block, flags=re.M)
        if head and row:
            out[IMPS[int(head.group(1))]] = {"ms_per_token": float(row.group(2)), "tokens": int(row.group(3)),
                                             "total_ms": float(row.group(1))}
    return out


def plot(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = summary["kernels"]
    e = summary["e2e"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = [n.replace("multithreading", "mt").replace("_programming", "").replace("loop_unrolling", "unroll")
              for n in IMPS]
    o3 = summary.get("O3", {})
    wd = 0.4 if o3 else 0.8
    xs = [i - wd / 2 if o3 else i for i in range(len(IMPS))]
    axes[0].bar(xs, [k[n]["gops"] for n in IMPS], width=wd, color="#2c5f8a", label="-O0 (course Makefile)")
    if o3:
        axes[0].bar([i + wd / 2 for i in range(len(IMPS))], [o3["kernels"][n]["gops"] for n in IMPS], width=wd,
                    color="#7fa7c9", label="-O3")
        axes[0].legend(fontsize=7)
    axes[0].set_ylabel("GOP/s (test_linear 4096x4096)")
    axes[0].set_yscale("log")
    have = [n for n in IMPS if n in e]
    axes[1].bar([IMPS.index(n) - (wd / 2 if o3 else 0) for n in have], [1000 / e[n]["ms_per_token"] for n in have],
                width=wd, color="#8a4f2c", label="-O0")
    e3 = o3.get("e2e", {})
    if e3:
        axes[1].bar([IMPS.index(n) + wd / 2 for n in e3], [1000 / v["ms_per_token"] for v in e3.values()], width=wd,
                    color="#d1a07f", label="-O3")
        axes[1].legend(fontsize=7)
    axes[1].set_ylabel("LLaMA2-7B decode tokens/s")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(range(len(IMPS)), labels, rotation=25, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "lab5_kernels.png", dpi=150)
    plt.close(fig)


def _read(name: str) -> str | None:
    p = RAW / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    kernels = parse_kernels(_read("test_linear.log"))
    e2e = parse_e2e(_read("e2e.log") or "")
    ref = kernels["reference"]["gops"]
    summary = {"host": "Intel Xeon Platinum 8470Q (AutoDL), g++ 11.4, -O0 -mavx2 -mfma",
               "upstream_commit": "f241215", "kernels": kernels, "e2e": e2e,
               "all_passed": all(v["passed"] for v in kernels.values()),
               "speedup_vs_reference": {n: kernels[n]["gops"] / ref for n in IMPS}}
    if "reference" in e2e:
        r = e2e["reference"]["ms_per_token"]
        summary["e2e_speedup_vs_reference"] = {n: r / v["ms_per_token"] for n, v in e2e.items()}
    if _read("test_linear_O3.log"):
        k3 = parse_kernels(_read("test_linear_O3.log"))
        summary["O3"] = {"kernels": k3, "all_passed": all(v["passed"] for v in k3.values()),
                         "speedup_vs_reference": {n: k3[n]["gops"] / k3["reference"]["gops"] for n in IMPS},
                         "e2e": parse_e2e(_read("e2e_O3.log") or "")}
    write_json(RAW / "summary.json", summary)
    plot(summary)
    print(summary["speedup_vs_reference"], summary.get("e2e_speedup_vs_reference"))


if __name__ == "__main__":
    main()
