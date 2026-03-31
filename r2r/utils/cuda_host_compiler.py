"""Normalize CC/CXX for CUDA extension JIT (FlashInfer, torch cpp extensions).

Conda environments often set CC/CXX to a very new GCC whose libstdc++ headers
break older nvcc. Call ensure_cuda_host_compiler_for_jit() at process start
(before spawning workers or importing SGLang) when appropriate.
"""

from __future__ import annotations

import os


def ensure_cuda_host_compiler_for_jit() -> None:
    """Set CC/CXX to /usr/bin/gcc and /usr/bin/g++ when conda or unset.

    Skip entirely if R2R_SKIP_CUDA_HOST_COMPILER_FIX is 1/true/yes.
    Does not override CC/CXX that are already set to a non-conda path.
    """
    v = os.environ.get("R2R_SKIP_CUDA_HOST_COMPILER_FIX", "").lower()
    if v in ("1", "true", "yes"):
        return
    gcc, gxx = "/usr/bin/gcc", "/usr/bin/g++"
    if not (os.path.isfile(gcc) and os.path.isfile(gxx)):
        return
    cc = os.environ.get("CC", "")
    if (not cc) or ("conda" in cc):
        os.environ["CC"] = gcc
    cxx = os.environ.get("CXX", "")
    if (not cxx) or ("conda" in cxx):
        os.environ["CXX"] = gxx
