# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and ctypes bindings of GenEPLB's share solvers (standard library only: no torch, no tensorrt_llm).

``csrc/l3_core.c`` is the rows ("C") split, ``csrc/l3b_core.c`` the cost-model ("B") split; both are deterministic,
single-threaded C, built with ``-O2 -ffp-contract=off`` (no FMA contraction, no fast-math) so that every
floating-point operation is evaluated as written. The B library links ``l3_core.c`` too: B's
candidate S_C is the rows split itself.

Each library is compiled once per (sources, flags, machine) key. The cache directory (often on a file system
shared by several nodes) is only read or added to: a process loads its node-local copy of an entry, a missing
entry is compiled node-locally and then published with a hard link that never replaces an existing file.
"""

import ctypes
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union

CSRC = Path(__file__).resolve().parent / "csrc"
ROWS_SOURCES = ("l3_core.c",)
B_SOURCES = ("l3b_core.c", "l3_core.c")
# the solvers' build line and build timeout
CFLAGS = ("-O2", "-ffp-contract=off", "-fPIC", "-shared", "-std=c11")
BUILD_TIMEOUT_S = 120
CACHE_ENV = "TRTLLM_GENEPLB_CACHE_DIR"
N_PAR, N_DIAG = 20, 24  # l3b_core.c N_PAR / N_DIAG (l3b_layout); P5: + act_k, act_var, 2 reserved

_LOCK = threading.Lock()
_LIBS = {}  # (library stem, cache dir) -> ctypes.CDLL: one build key and load per process


def default_cache_dir() -> Path:
    """``$TRTLLM_GENEPLB_CACHE_DIR`` or ``~/.cache/tensorrt_llm/geneplb``."""
    return Path(os.environ.get(CACHE_ENV) or os.path.expanduser("~/.cache/tensorrt_llm/geneplb"))


def local_dir() -> Path:
    """This node's private directory for the copies that are loaded (never on a shared file system)."""
    return Path(tempfile.gettempdir()) / ("tensorrt_llm_geneplb_%d" % os.getuid())


def fetch(shared: Path, local: Path) -> bool:
    """Copy a cache entry to its node-local path once (atomic rename); False if neither has it. A node-local copy
    the cache lacks (e.g. a fresh cache directory) is published to it."""
    if local.exists():
        if not shared.exists():
            publish(local, shared)
        return True
    if not shared.exists():
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(".%s.%d.%d" % (local.name, os.getpid(), threading.get_ident()))
    shutil.copyfile(shared, tmp)
    os.replace(tmp, local)
    return True


def publish(local: Path, shared: Path) -> None:
    """Best effort: add a node-local build to the cache. A hard link never replaces an entry that another node
    may have loaded; a lost race or a read-only cache leaves the cache as it is."""
    tmp = shared.with_name(".%s.%s.%d" % (shared.name, socket.gethostname(), os.getpid()))
    try:
        shared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, tmp)
        os.link(tmp, shared)
    except OSError:
        pass
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def build_key(sources) -> str:
    """sha256 over the sources' bytes, the flags and the machine: a changed source or flag never reuses a build."""
    h = hashlib.sha256()
    for name in sources:
        h.update(name.encode() + b"\0" + (CSRC / name).read_bytes() + b"\0")
    h.update(repr(CFLAGS).encode())
    h.update(platform.machine().encode())
    return h.hexdigest()


def _build(cache_dir: Optional[Union[str, Path]], stem: str, sources) -> Path:
    """The node-local copy of ``<cache_dir>/solvers/<stem>-<key>.so``, compiled here and published if missing."""
    name = "%s-%s.so" % (stem, build_key(sources)[:16])
    shared = (
        Path(cache_dir if cache_dir is not None else default_cache_dir()).expanduser().resolve()
        / "solvers"
        / name
    )
    local = local_dir() / "solvers" / name
    if fetch(shared, local):
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(".%s.%d.%d" % (name, os.getpid(), threading.get_ident()))
    cmd = ["gcc", *CFLAGS, "-o", str(tmp), *[str(CSRC / src) for src in sources], "-lm"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
        if done.returncode != 0:
            raise RuntimeError(
                "GenEPLB solver build failed (%s):\n%s" % (" ".join(cmd), done.stderr[-4000:])
            )
        os.replace(tmp, local)  # concurrent builders of this node never see a partial file
    finally:
        if tmp.exists():
            tmp.unlink()
    publish(local, shared)
    return local


def _bind_rows(lib: ctypes.CDLL) -> None:
    ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    D, Ci = ctypes.c_double, ctypes.c_int
    # int l3_weights_full(E, R, LS, load, n_slots, phys, align_tile, tile, w, T_out, improved, mode, g, max_newton,
    #   stats)
    lib.l3_weights_full.argtypes = [
        Ci,
        Ci,
        Ci,
        dp,
        Ci,
        ip,
        Ci,
        D,
        dp,
        dp,
        ctypes.POINTER(Ci),
        Ci,
        D,
        Ci,
        ip,
    ]
    lib.l3_weights_full.restype = Ci


def build_rows_solver(cache_dir: Optional[Union[str, Path]] = None) -> ctypes.CDLL:
    """The rows ("C") solver library: ``l3_weights_full`` with its ctypes argtypes."""
    with _LOCK:
        lib = _LIBS.get(("l3_core", str(cache_dir)))
        if lib is None:
            lib = ctypes.CDLL(str(_build(cache_dir, "l3_core", ROWS_SOURCES)))
            _bind_rows(lib)
            _LIBS[("l3_core", str(cache_dir))] = lib
        return lib


def build_b_solver(cache_dir: Optional[Union[str, Path]] = None) -> ctypes.CDLL:
    """The cost-model ("B") solver library (``l3b_core.c`` + ``l3_core.c``) with its ctypes argtypes; checks
    its layout."""
    with _LOCK:
        lib = _LIBS.get(("l3b_core", str(cache_dir)))
        if lib is not None:
            return lib
        lib = ctypes.CDLL(str(_build(cache_dir, "l3b_core", B_SOURCES)))
        ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
        D, Ci = ctypes.c_double, ctypes.c_int
        # int l3b_weights(E, R, LS, load, phys, par, w_out, diag_out)
        lib.l3b_weights.argtypes = [Ci, Ci, Ci, dp, ip, dp, dp, dp]
        lib.l3b_weights.restype = Ci
        _bind_rows(lib)
        lib.l3b_det_exp.argtypes = [D]
        lib.l3b_det_exp.restype = D
        lib.l3b_det_log.argtypes = [D]
        lib.l3b_det_log.restype = D
        lib.l3b_erfc.argtypes = [D]
        lib.l3b_erfc.restype = D
        lib.l3b_qexp.argtypes = [D, D, D]
        lib.l3b_qexp.restype = D
        lib.l3b_lse.argtypes = [dp, Ci, D]
        lib.l3b_lse.restype = D
        lib.l3b_tables.argtypes = [dp, dp, dp, dp]
        lib.l3b_tables.restype = None
        lib.l3b_layout.argtypes = [ip, ip]
        lib.l3b_layout.restype = Ci
        n_par, n_diag = Ci(), Ci()
        lib.l3b_layout(ctypes.byref(n_par), ctypes.byref(n_diag))
        if (n_par.value, n_diag.value) != (N_PAR, N_DIAG):
            raise RuntimeError(
                "l3b_core.c layout %d / %d != %d / %d" % (n_par.value, n_diag.value, N_PAR, N_DIAG)
            )
        _LIBS[("l3b_core", str(cache_dir))] = lib
        return lib
