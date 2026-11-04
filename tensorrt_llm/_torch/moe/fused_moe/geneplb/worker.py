# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The planner worker: one plain Python process per rank, so the decisions never hold the serving
thread's GIL. It imports only numpy and this directory's standalone modules (never torch or
tensorrt_llm) and answers length-prefixed pickled requests on stdin with replies on stdout.
"""

import os
import pickle
import struct
import subprocess
import sys

_HEADER = struct.Struct("<Q")


def _write(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    view = memoryview(_HEADER.pack(len(data)) + data)
    while view:  # a raw pipe may take part of a large write
        n = stream.write(view)
        view = view[n:]
    stream.flush()


def _read_exact(stream, n: int):
    chunks, left = [], n
    while left:
        chunk = stream.read(left)
        if not chunk:
            return None
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def _read(stream):
    head = _read_exact(stream, _HEADER.size)
    if head is None:
        return None
    body = _read_exact(stream, _HEADER.unpack(head)[0])
    return None if body is None else pickle.loads(body)


def serve() -> None:
    """Worker side: (token, request) in, (token, reply | exception text) out, until EOF."""
    import planner
    import shares

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing but replies on the pipe
    while True:
        msg = _read(stdin)
        if msg is None:
            return
        token, kind, payload = msg
        try:
            if kind == "init":
                shares.configure(payload["cache_dir"])
                reply = True
            else:
                reply = planner.decide(payload)
            _write(stdout, (token, reply, None))
        except Exception as exc:  # reported to the serving process, which raises it
            _write(stdout, (token, None, f"{type(exc).__name__}: {exc}"))


class Worker:
    """Serving side: start the worker, send requests, poll replies without blocking."""

    def __init__(self, cache_dir: str):
        here = os.path.dirname(os.path.abspath(__file__))
        boot = "import sys; sys.path.insert(0, %r); import worker; worker.serve()" % here
        self._proc = subprocess.Popen(
            [sys.executable, "-s", "-c", boot],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self._token = 0
        self._pending = 0
        self.request("init", dict(cache_dir=cache_dir))
        self.wait()

    def request(self, kind: str, payload) -> int:
        self._token += 1
        _write(self._proc.stdin, (self._token, kind, payload))
        self._pending += 1
        return self._token

    def poll(self):
        """The next reply if one is ready, else None."""
        import select

        if not self._pending:
            return None
        ready, _, _ = select.select([self._proc.stdout], [], [], 0)
        return self._take() if ready else None

    def wait(self):
        return self._take()

    def _take(self):
        msg = _read(self._proc.stdout)
        if msg is None:
            raise RuntimeError(f"GenEPLB planner worker exited (code {self._proc.poll()})")
        self._pending -= 1
        token, reply, error = msg
        if error is not None:
            raise RuntimeError(f"GenEPLB planner worker: {error}")
        return token, reply

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
