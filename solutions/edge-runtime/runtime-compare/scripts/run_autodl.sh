#!/bin/bash
# The 2026-11-12 run on an AutoDL no-GPU container (0.5 core, 2 GB cgroup). Run from the repo root.
# Three venvs, because the runtimes' torch pins conflict:
#   rtvenv  : torch 2.14.1+cpu, litert-torch 0.10.0, ai-edge-litert 2.3.0
#   etvenv  : torch 2.15.0+cpu, executorch 1.6.0 (pins torch==2.15.0)
#   tvmvenv : apache-tvm 0.27.0.post1, apache-tvm-ffi 0.1.14.post0, plus a sitecustomize.py that sets
#             sys.modules["torch"] = None (TVM JIT-builds a torch addon whenever torch imports)
# Torch wheels come from https://download.pytorch.org/whl/cpu: the PyPI 2.15 wheel needs CUDA 13.
set -u
V=${VENVS:-/root/autodl-tmp}
export OMP_NUM_THREADS=1 TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export RTCMP_PY_LITERT=$V/rtvenv/bin/python
export RTCMP_PY_EXECUTORCH=$V/etvenv/bin/python
export RTCMP_PY_TVM=$V/tvmvenv/bin/python
export RTCMP_EVICT=$V:$(dirname "$(dirname "$(command -v python)")")
L=solutions/edge-runtime/runtime-compare
rm -f $L/artifacts/raw/part-*.json
python $L/scripts/run_experiments.py
