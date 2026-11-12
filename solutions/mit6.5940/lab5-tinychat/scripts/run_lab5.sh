#!/usr/bin/env bash
# Reproduce Lab 5 on a Linux x86 host with AVX2: apply the patch to tinychat-tutorial f241215,
# run the official evaluate.sh (correctness + GOPs of the six kernels), then decode 32 tokens of
# LLaMA2-7B-chat with each kernel. Logs land in ../artifacts/raw/.
#
#   bash scripts/run_lab5.sh /path/to/workdir [cores, default 8-15] [e2e cores, default 16-21]
set -euo pipefail
LAB=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:?workdir}
CORES=${2:-8-15}
E2E_CORES=${3:-16-21}
mkdir -p "$WORK" "$LAB/artifacts/raw"
cd "$WORK"
[ -d tinychat-tutorial ] || git clone --recursive https://github.com/mit-han-lab/tinychat-tutorial
cd tinychat-tutorial
git checkout -q f241215
git apply --check "$LAB/patch/x86.patch" 2>/dev/null && git apply "$LAB/patch/x86.patch"
cd transformer
if [ ! -d INT4/models/LLaMA_7B_2_chat ]; then
  # download_model.py points at the same Hugging Face file; set HF_ENDPOINT for a mirror.
  python download_model.py --model LLaMA2_7B_chat_awq_int4 --QM QM_x86
fi
taskset -c "$CORES" bash evaluate.sh 2>&1 | grep -a -E 'Sanity|Section|, [0-9.]+, [0-9.]+, 10,' > "$LAB/artifacts/raw/test_linear.log"

# End to end: a copy of chat.cc that reads N_PREDICT and exits on EOF; not part of the patch.
B="$WORK/tc_bench"
rm -rf "$B" && mkdir -p "$B"
(cd .. && git ls-files | tar cf - -T - | tar xf - -C "$B" && cp kernels/starter_code/*.cc "$B/kernels/starter_code/")
ln -s "$PWD/INT4" "$B/transformer/INT4"
cd "$B/transformer"
sed -i 's/generation_config.n_predict = 512;/generation_config.n_predict = getenv("N_PREDICT") ? atoi(getenv("N_PREDICT")) : 512;/' application/chat.cc
sed -i 's/^\(\s*\)std::getline(std::cin, input);/\1std::getline(std::cin, input); if (std::cin.eof() \&\& input.empty()) return 0;/' application/chat.cc
: > "$LAB/artifacts/raw/e2e.log"
for imp in 5 4 3 2 1 0; do
  make clean >/dev/null 2>&1
  make chat -j 8 IMP=$imp >/dev/null 2>&1
  echo "=== IMP $imp" >> "$LAB/artifacts/raw/e2e.log"
  printf 'What is the capital of France? Answer in one paragraph.\n' | N_PREDICT=32 taskset -c "$E2E_CORES" ./chat 2>&1 \
    | grep -a 'Inference latency' >> "$LAB/artifacts/raw/e2e.log"
done

# Optional: the same code with -O3 instead of the course's -O0 (kernels, then reference and all_techniques end to end).
if [ "${O3:-0}" = 1 ]; then
  O="$WORK/tc_o3"
  rm -rf "$O" && cp -a "$B" "$O" && sed -i 's/-O0/-O3/' "$O/transformer/Makefile"
  cd "$O/transformer"
  taskset -c "$CORES" bash evaluate.sh 2>&1 | grep -a -E 'Sanity|Section|, [0-9.]+, [0-9.]+, 10,' > "$LAB/artifacts/raw/test_linear_O3.log"
  : > "$LAB/artifacts/raw/e2e_O3.log"
  for imp in 5 0; do
    make clean >/dev/null 2>&1
    make chat -j 8 IMP=$imp >/dev/null 2>&1
    echo "=== IMP $imp" >> "$LAB/artifacts/raw/e2e_O3.log"
    printf 'What is the capital of France? Answer in one paragraph.\n' | N_PREDICT=32 taskset -c "$E2E_CORES" ./chat 2>&1 \
      | grep -a 'Inference latency' >> "$LAB/artifacts/raw/e2e_O3.log"
  done
fi
python "$LAB/scripts/run_experiments.py"
