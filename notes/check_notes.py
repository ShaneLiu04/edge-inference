"""Strict validator for the edge-deploy module notes (same contract as mlsys-infra).

Checks performed
----------------
1. All 8 module files exist.
2. Every note has the required metadata and section structure, including `## 深入推导`.
3. Every note contains at least three fenced formula/code blocks.
4. Every note has an interview follow-up chain, self-test, and >=2 cross links,
   one of which points at the matching mlsys-infra module.
5. Relative links resolve on disk; mlsys-infra links resolve against a sibling checkout when present.
6. notes/README.md indexes every module; GLOSSARY/ROADMAP exist and cover
   the required key terms.
7. No TODO/TBD placeholders remain.

Usage
-----
    python notes/check_notes.py            # concise summary
    python notes/check_notes.py --verbose  # per-note table
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODULES = {
    "module-01-edge-roofline.md": "M01",
    "module-02-integer-contract.md": "M02",
    "module-03-calibration-localization.md": "M03",
    "module-04-static-memory-planning.md": "M04",
    "module-05-convolution-kernels.md": "M05",
    "module-06-schedule-search.md": "M06",
    "module-07-dma-double-buffering.md": "M07",
    "module-08-block-quantized-decode.md": "M08",
}

REQUIRED_SECTIONS = [
    "## 摘要",
    "## 一页速览卡",
    "## 学习目标",
    "## 先修知识",
    "## 核心概念",
    "## 关键推导",
    "## 深入推导",
    "## 白板估算",
    "## 实现映射",
    "## 易错点与反例",
    "## 实验或练习",
    "## 高频问题",
    "## 面试追问链",
    "## 自测题",
    "## 关联笔记",
    "## 本讲小结",
    "## 延伸阅读",
]

REQUIRED_METADATA = ["- 课程出处", "- 对应实验", "- 实测结论", "- 状态"]

MIN_LINES = 180
MIN_FORMULA_BLOCKS = 3
MIN_RELATED_LINKS = 2
MIN_FOLLOWUPS = 5
MLSYS_URL = "https://gitee.com/liu-xingyan04/mlsys-infra/blob/main/notes/module-"

KEY_GLOSSARY_TERMS = [
    "roofline",
    "算术强度",
    "ridge point",
    "VNNI",
    "requantization",
    "zero-point",
    "per-channel",
    "SRDHM",
    "round-half-even",
    "calibration",
    "KL",
    "QAT",
    "STE",
    "BN folding",
    "arena",
    "liveness",
    "in-place",
    "patch-based",
    "im2col",
    "Winograd",
    "depthwise",
    "tiling",
    "cost model",
    "double buffering",
    "DMA",
    "scratchpad",
    "Q4_0",
    "Q8_0",
    "GGUF",
    "perplexity",
    "decode",
]

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#\s]+)(?:#[^)]*)?\)")
FENCE_RE = re.compile(r"^```(?:text|python|bash|cpp|c)\b", re.IGNORECASE | re.MULTILINE)
FOLLOWUP_RE = re.compile(r"^\s*\d+\.\s*问[:：]", re.MULTILINE)
FORBIDDEN = re.compile(r"^\s*[-*#>]*\s*(TODO|TBD)\s*[:：]?\s*$|待补|待写", re.MULTILINE)


def section(text: str, header: str) -> str:
    if header not in text:
        return ""
    return text.split(header, 1)[1].split("\n## ", 1)[0]


def check_note(path: Path) -> tuple[dict[str, int], list[str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    issues: list[str] = []

    for header in REQUIRED_SECTIONS:
        if header not in text:
            issues.append(f"缺少章节 {header}")
    for meta in REQUIRED_METADATA:
        if meta not in text:
            issues.append(f"缺少元信息 {meta}")
    if len(lines) < MIN_LINES:
        issues.append(f"行数 {len(lines)} < {MIN_LINES}")

    fences = len(FENCE_RE.findall(text))
    if fences < MIN_FORMULA_BLOCKS:
        issues.append(f"公式/代码块 {fences} < {MIN_FORMULA_BLOCKS}")

    followups = len(FOLLOWUP_RE.findall(section(text, "## 面试追问链")))
    if followups < MIN_FOLLOWUPS:
        issues.append(f"面试追问 {followups} < {MIN_FOLLOWUPS}")

    related_text = section(text, "## 关联笔记")
    related = [t for t in LINK_RE.findall(related_text) if t.endswith(".md")]
    if len(related) < MIN_RELATED_LINKS:
        issues.append(f"关联笔记链接 {len(related)} < {MIN_RELATED_LINKS}")
    if MLSYS_URL not in related_text:
        issues.append("关联笔记缺少 mlsys-infra 对应模块链接")

    hit = FORBIDDEN.search(text)
    if hit:
        issues.append(f"存在占位符 {hit.group(0)!r}")

    sibling = path.parents[2] / "mlsys-infra" / "notes"
    for target in LINK_RE.findall(text):
        if target.startswith(MLSYS_URL) and sibling.is_dir():
            name = target.rsplit("/", 1)[1]
            if not (sibling / name).exists():
                issues.append(f"mlsys-infra 无此笔记 {name}")
            continue
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).resolve().exists():
            issues.append(f"失效链接 {target}")

    stats = {"lines": len(lines), "fences": fences, "followups": followups,
             "questions": text.count("?") + text.count("？")}
    return stats, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print per-note stats")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    issues: list[str] = []

    if args.verbose:
        print(f"{'module':<46}{'lines':>6}{'fences':>8}{'chain':>7}{'q':>6}")
        print("-" * 73)

    for name, label in MODULES.items():
        path = root / name
        if not path.exists():
            issues.append(f"{label}: 文件缺失 {name}")
            continue
        stats, note_issues = check_note(path)
        if args.verbose:
            print(f"{name:<46}{stats['lines']:>6}{stats['fences']:>8}{stats['followups']:>7}{stats['questions']:>6}")
        issues.extend(f"{label}: {item}" for item in note_issues)

    readme = root / "README.md"
    readme_text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if not readme_text:
        issues.append("notes/README.md 缺失")
    for name in MODULES:
        if readme_text and name not in readme_text:
            issues.append(f"notes/README.md 未索引 {name}")

    glossary = root / "GLOSSARY.md"
    if not glossary.exists():
        issues.append("GLOSSARY.md 缺失")
    else:
        g = glossary.read_text(encoding="utf-8").lower()
        missing = [t for t in KEY_GLOSSARY_TERMS if t.lower() not in g]
        if missing:
            issues.append(f"GLOSSARY 缺少关键术语: {', '.join(missing)}")

    if not (root / "READING-ROADMAP.md").exists():
        issues.append("READING-ROADMAP.md 缺失")

    if issues:
        print("\n".join(f"  - {item}" for item in issues))
        print(f"\nFAILED: {len(issues)} issue(s)")
        return 1

    print(f"OK: {len(MODULES)} notes, structure / formulas / links / index / glossary verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
