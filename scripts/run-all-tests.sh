#!/usr/bin/env bash
# Run every test suite on Linux; same suites as run-all-tests.ps1.
# Usage: PYTHON=python3 ./scripts/run-all-tests.sh   (suites run in parallel, one pytest each)
set -u
PY=${PYTHON:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
SUITES=(
  "tools edge_calc + harness|tools/tests"
  "edge-runtime host-roofline|solutions/edge-runtime/host-roofline/tests"
  "edge-runtime int8-kernels|solutions/edge-runtime/int8-kernels/tests"
  "edge-runtime conv-kernels|solutions/edge-runtime/conv-kernels/tests"
  "edge-runtime dma-double-buffer|solutions/edge-runtime/dma-double-buffer/tests"
  "edge-runtime q4-gemv|solutions/edge-runtime/q4-gemv/tests"
  "edge-runtime llama-cpp-7b|solutions/edge-runtime/llama-cpp-7b/tests"
  "edge-runtime runtime-compare|solutions/edge-runtime/runtime-compare/tests"
  "mit6.5940 lab1-pruning|solutions/mit6.5940/lab1-pruning/tests"
  "mit6.5940 lab2-quantization|solutions/mit6.5940/lab2-quantization/tests"
  "mit6.5940 lab2-vgg|solutions/mit6.5940/lab2-vgg/tests"
  "mit6.5940 lab3-nas|solutions/mit6.5940/lab3-nas/tests"
  "mit6.5940 lab4-awq|solutions/mit6.5940/lab4-awq/tests"
  "mit6.5940 lab5-tinychat|solutions/mit6.5940/lab5-tinychat/tests"
  "mit6.5940 tinyengine-memory|solutions/mit6.5940/tinyengine-memory/tests"
  "mit6.5940 project-mcu-e2e|solutions/mit6.5940/project-mcu-e2e/tests"
  "mlc tensor-program-tuning|solutions/mlc/tensor-program-tuning/tests"
  "mlc e2e-relax|solutions/mlc/e2e-relax/tests"
)
TMP=$(mktemp -d)
JOBS=$(nproc)
run_suite() {
  local i=$1 dir=$2
  ( cd "$(dirname "$dir")" && OMP_NUM_THREADS=1 "$PY" -m pytest -q -p no:cacheprovider "$(basename "$dir")" > "$TMP/$i.out" 2>&1; echo $? > "$TMP/$i.rc" )
}
i=0
for s in "${SUITES[@]}"; do
  run_suite $i "$ROOT/${s#*|}" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 0.2; done
  i=$((i+1))
done
wait
fail=0
i=0
for s in "${SUITES[@]}"; do
  name=${s%%|*}
  rc=$(cat "$TMP/$i.rc"); summary=$(grep -E "passed|failed|error" "$TMP/$i.out" | tail -1)
  if [ "$rc" = 0 ]; then res=PASS; else res=FAIL; fail=1; tail -30 "$TMP/$i.out"; fi
  printf "%-34s %-5s %s\n" "$name" "$res" "$summary"
  i=$((i+1))
done
rm -rf "$TMP"
exit $fail
