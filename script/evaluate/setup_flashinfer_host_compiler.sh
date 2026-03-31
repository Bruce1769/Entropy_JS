#!/usr/bin/env bash
# FlashInfer JIT passes $CC to nvcc as -ccbin. Conda's activate often sets CC/CXX to
# a very new GCC whose libstdc++ uses C++23 bfloat16 literals that nvcc cannot parse.
# Run AFTER `conda activate <env>`:
#   source script/evaluate/setup_flashinfer_host_compiler.sh
#
# Override if your distro uses another compiler:
#   export R2R_FLASHINFER_HOST_CXX=/usr/bin/g++-12

_host="${R2R_FLASHINFER_HOST_CXX:-/usr/bin/g++-11}"
if [[ ! -x "$_host" ]]; then
  echo "setup_flashinfer_host_compiler: $_host not executable; set R2R_FLASHINFER_HOST_CXX" >&2
  return 1 2>/dev/null || exit 1
fi
export CC="$_host"
export CXX="$_host"
export CUDAHOSTCXX="$_host"
