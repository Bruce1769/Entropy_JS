"""
Ensure CUDA_HOME / PATH / CC / CXX are valid for FlashInfer, Triton, and SGLang JIT.

Conda envs often have PyTorch + pip CUDA wheels but no `nvcc`. Build systems then fall back to
``/usr/local/cuda``, which may not exist. We detect ``nvcc`` next to the active env or in the
parent Miniconda/Anaconda install (``.../miniconda3/bin/nvcc``) and export ``CUDA_HOME`` accordingly.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


def _ensure_nvcc_tmpdir() -> None:
    """
    nvcc / FlashInfer JIT write temp files under TMPDIR (default often ``/tmp``).
    On shared clusters ``/tmp`` may be full, quota-limited, or unwritable →
    ``nvcc fatal: Could not open output file '/tmp/tmpxft_...'``.

    A single ``mkstemp`` probe can succeed on ``/tmp`` while parallel ``nvcc --threads=32``
    still fails (space / inode exhaustion). So whenever the effective temp directory
    resolves to ``/tmp``, we redirect to ``~/.cache/r2r_jit_tmp`` unless the user opted
    out with ``R2R_USE_SYSTEM_TMP_ONLY=1`` or set ``TMPDIR`` to a path that does not
    resolve to ``/tmp``.

    **Important:** If ``TMPDIR`` is unset, ``nvcc`` still defaults to ``/tmp``, while
    Python's ``tempfile.gettempdir()`` may fall back to the current working directory
    when ``/tmp`` fails its probe. That mismatch made us skip setting ``TMPDIR`` and
    left nvcc writing to a broken ``/tmp``. When ``TMPDIR`` is unset, we always set
    ``TMPDIR``/``TEMP``/``TMP`` to ``~/.cache/r2r_jit_tmp``.

    Overrides: ``R2R_JIT_TMPDIR`` (directory path), ``R2R_RESPECT_TMPDIR=1`` (do not
    change an already-set ``TMPDIR``), ``R2R_USE_SYSTEM_TMP_ONLY=1`` (disable all of this).
    """
    if os.environ.get("R2R_USE_SYSTEM_TMP_ONLY"):
        return

    def _dir_allows_tempfile(directory: str) -> bool:
        try:
            fd, path = tempfile.mkstemp(prefix="r2r_nvcc_probe_", dir=directory)
            os.close(fd)
            os.unlink(path)
            return True
        except OSError:
            return False

    def _apply_tmpdir(path: str) -> bool:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        if not _dir_allows_tempfile(path):
            return False
        os.environ["TMPDIR"] = path
        os.environ["TEMP"] = path
        os.environ["TMP"] = path
        return True

    override = os.environ.get("R2R_JIT_TMPDIR")
    if override:
        _apply_tmpdir(os.path.expanduser(override))
        return

    tmpdir_env = os.environ.get("TMPDIR")
    if tmpdir_env and os.environ.get("R2R_RESPECT_TMPDIR"):
        return

    # nvcc defaults to /tmp when TMPDIR is unset — do not trust Python's gettempdir()
    # (it may be cwd if /tmp failed Python's probe).
    if not tmpdir_env:
        for fb in (
            str(Path.home() / ".cache" / "r2r_jit_tmp"),
            str(Path.home() / ".r2r_jit_tmp"),
        ):
            if _apply_tmpdir(fb):
                return
        return

    candidate = tmpdir_env
    try:
        resolved = os.path.realpath(candidate)
    except OSError:
        resolved = os.path.abspath(candidate)

    tmp_root = os.path.realpath("/tmp")
    # Avoid shared /tmp for nvcc JIT: parallel compiles need many large temp files.
    use_home_tmp = resolved == tmp_root

    if not use_home_tmp:
        if _dir_allows_tempfile(candidate):
            return
        use_home_tmp = True

    fallback = str(Path.home() / ".cache" / "r2r_jit_tmp")
    if not _apply_tmpdir(fallback):
        _apply_tmpdir(str(Path.home() / ".r2r_jit_tmp"))


def _conda_root_with_nvcc(conda_prefix: str) -> Optional[str]:
    """If CONDA_PREFIX is .../envs/<name>, return install root when root/bin/nvcc exists."""
    if not conda_prefix:
        return None
    p = os.path.abspath(conda_prefix)
    while True:
        if os.path.basename(p) == "envs":
            root = os.path.dirname(p)
            nvcc = os.path.join(root, "bin", "nvcc")
            return root if os.path.isfile(nvcc) else None
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def _first_existing_nvcc() -> Optional[str]:
    """Return absolute path to nvcc if found."""
    seen: set[str] = set()

    def consider(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        path = os.path.abspath(path)
        if path in seen:
            return None
        seen.add(path)
        return path if os.path.isfile(path) else None

    # Explicit env (user override)
    for key in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(key)
        if root:
            hit = consider(os.path.join(root, "bin", "nvcc"))
            if hit:
                return hit

    which = shutil.which("nvcc")
    hit = consider(which)
    if hit:
        return hit

    prefix = os.environ.get("CONDA_PREFIX")
    for candidate in (os.path.join(prefix, "bin", "nvcc") if prefix else None,):
        hit = consider(candidate)
        if hit:
            return hit
    if prefix:
        root = _conda_root_with_nvcc(prefix)
        if root:
            hit = consider(os.path.join(root, "bin", "nvcc"))
            if hit:
                return hit

    return None


def _ensure_lib64_has_libcudart(cuda_home: str) -> None:
    """
    FlashInfer's ninja template always passes ``-L$cuda_home/lib64`` before ``-lcudart``.
    Many Conda installs only ship ``libcudart`` under ``lib/``, not ``lib64/``, which breaks
    the link step. If ``cuda_home`` is writable, add ``lib64/libcudart.so -> ../lib/libcudart.so``.
    """
    root = Path(cuda_home)
    lib = root / "lib"
    lib64 = root / "lib64"
    src = lib / "libcudart.so"
    if not src.is_file():
        return
    dst = lib64 / "libcudart.so"
    try:
        lib64.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if dst.is_file() or dst.is_symlink():
        return
    try:
        dst.symlink_to(os.path.relpath(src, lib64))
    except OSError:
        pass


def _prepend_unique_path_list(key: str, front_dirs: list[str]) -> None:
    """Prepend directories to a PATH-style env var (linker / loader search paths)."""
    parts: list[str] = []
    for d in front_dirs:
        if d and os.path.isdir(d) and d not in parts:
            parts.append(d)
    existing = os.environ.get(key, "")
    for p in existing.split(os.pathsep) if existing else []:
        if p and p not in parts:
            parts.append(p)
    os.environ[key] = os.pathsep.join(parts)


def ensure_cuda_jit_environment() -> None:
    """
    Idempotent: set CUDA_HOME, prepend CUDA bin to PATH, and prefer conda compilers when present.
    Safe to call before importing torch/sglang.
    """
    _ensure_nvcc_tmpdir()

    nvcc = _first_existing_nvcc()
    cuda_home: Optional[str] = None
    if nvcc:
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
        os.environ["CUDA_HOME"] = cuda_home
        # Some NVIDIA tooling reads CUDA_PATH instead of CUDA_HOME
        os.environ.setdefault("CUDA_PATH", cuda_home)
        _ensure_lib64_has_libcudart(cuda_home)
        bin_dir = os.path.join(cuda_home, "bin")
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + path

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        cc = os.path.join(prefix, "bin", "x86_64-conda-linux-gnu-cc")
        cxx = os.path.join(prefix, "bin", "x86_64-conda-linux-gnu-c++")
        if os.path.isfile(cc):
            os.environ.setdefault("CC", cc)
        if os.path.isfile(cxx):
            os.environ.setdefault("CXX", cxx)

    # FlashInfer ninja links with -lcudart; Conda often puts libcudart in $CUDA_HOME/lib, not lib64.
    lib_front: list[str] = []
    for root in (cuda_home, prefix):
        if not root:
            continue
        for sub in ("lib", "lib64"):
            p = os.path.join(root, sub)
            if os.path.isdir(p):
                lib_front.append(p)
    if lib_front:
        _prepend_unique_path_list("LIBRARY_PATH", lib_front)
        _prepend_unique_path_list("LD_LIBRARY_PATH", lib_front)
