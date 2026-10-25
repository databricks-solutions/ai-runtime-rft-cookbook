# Derived from verl-on-air (Apache License 2.0), usecases/officeqa/sandbox.py at b088a42 (its parent
# side); modified for ai-runtime-rft-cookbook: sandbox levels, ToolResult output, the child as a
# separate script, the child's output read with a limit. Output text is unchanged. See
# THIRD_PARTY_NOTICES.md.
"""Run model-written Python in a fresh child interpreter, isolated by the kernel.

Levels:

    strict       bubblewrap: new user, PID, network, IPC and UTS namespaces, a read-only root, an
                 empty /proc, a private /tmp, data and credential directories masked by empty
                 tmpfs, a cleared environment. No network. If bubblewrap or user namespaces are
                 unavailable the call is an infrastructure failure: nothing runs.
    best_effort  strict when available, otherwise a child process with a cleared environment and
                 CPU, memory and file-size limits only (no filesystem or network boundary). The
                 fallback is logged once and recorded on every call.
    off          the plain child process; tests only.

Each call records the level it reached in `ToolResult.sandbox`. Every call is a fresh child, so
nothing persists between calls. The child imports only the standard library and the preloaded
calculation stack (numpy, pandas); the boundary is the namespace, not an import filter.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from rft_cookbook.tools.base import ToolInfraError, ToolResult

LEVELS = ("strict", "best_effort", "off")
# Data, credentials and scratch the child must not see: corpora, answer keys, model weights,
# workspace credentials and the code snapshot all live under these on AI Runtime nodes.
MASKED = ("/Volumes", "/dbfs", "/home", "/root", "/local_disk0", "/mnt", "/media", "/srv", "/databricks")
CHILD = Path(__file__).with_name("sandbox_child.py")
CHILD_IN_SANDBOX = "/tmp/_rft_sandbox_child.py"
STRICT_FEATURES = "bwrap:user,net,pid,ipc,uts,ro-root,empty-proc,private-tmp,masked,clearenv"
# What the parent reads of each of the child's streams: the child keeps its own output to a few
# times `max_output`, so only a child that writes to its file descriptors directly comes near this.
READ_LIMIT = 1 << 20
FALLBACK_FEATURES = "subprocess:clearenv,rlimits"

_probe_lock = threading.Lock()
_bwrap_ok: bool | None = None
_fallback_logged = False


def bwrap_available() -> bool:
    """bubblewrap exists and the runtime allows the namespaces it needs (cached)."""
    global _bwrap_ok
    with _probe_lock:
        if _bwrap_ok is None:
            binary = shutil.which("bwrap")
            if not binary:
                _bwrap_ok = False
            else:
                probe = [binary, "--unshare-user", "--unshare-net", "--unshare-pid", "--ro-bind", "/", "/",
                         "--tmpfs", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "true"]
                try:   # AI Runtime forbids mounting a fresh proc, so /proc is an empty tmpfs
                    _bwrap_ok = subprocess.run(probe, capture_output=True, timeout=15).returncode == 0
                except Exception:  # noqa: BLE001
                    _bwrap_ok = False
        return _bwrap_ok


class PythonSandbox:
    def __init__(self, level: str = "best_effort", *, timeout_s: float = 15.0, mem_mb: int = 0,
                 max_code: int = 20_000, max_output: int = 8_000, preload: tuple[str, ...] = ("numpy", "pandas")):
        if level not in LEVELS:
            raise ValueError(f"sandbox level {level!r}: one of {LEVELS}")
        self.level, self.timeout_s, self.mem_mb = level, float(timeout_s), int(mem_mb)
        self.max_code, self.max_output, self.preload = max_code, max_output, preload

    def run(self, code: str) -> ToolResult:
        code = str(code or "").strip()
        if not code:
            return ToolResult("Error: 'code' must be a non-empty string", status="error")
        if len(code) > self.max_code:
            return ToolResult(f"Error: code exceeds {self.max_code} chars", status="error")
        argv, env, reached = self._plan()
        try:
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=env, close_fds=True)
        except Exception as error:  # noqa: BLE001 - never fall back to running in-process
            raise ToolInfraError(f"compute sandbox failed to launch ({type(error).__name__}: {error})") from error
        out, err, timed_out = _communicate(process, code.encode("utf-8"), self.timeout_s)
        if timed_out:
            return ToolResult(self._clip(f"Error: TimeoutError: code execution exceeded {self.timeout_s:.0f}s"),
                              status="timeout", sandbox=reached)
        if len(out) > READ_LIMIT:
            return ToolResult(f"Error: compute wrote more than {READ_LIMIT} bytes of output", status="error",
                              sandbox=reached)
        text = self._render(out, err, process.returncode)
        return ToolResult(text, status="error" if text.startswith("Error:") else "ok", sandbox=reached)

    def _plan(self) -> tuple[list[str], dict[str, str], str]:
        if self.level != "off" and bwrap_available():
            return self._bwrap(), {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, STRICT_FEATURES
        if self.level == "strict":
            raise ToolInfraError("compute sandbox unavailable: bubblewrap or user namespaces are missing "
                                 "and the sandbox level is strict")
        global _fallback_logged
        if self.level == "best_effort" and not _fallback_logged:
            print("[sandbox] bubblewrap unavailable: compute runs in a resource-limited subprocess, "
                  "without a filesystem or network boundary", file=sys.stderr, flush=True)
            _fallback_logged = True
        env = {**self._child_env(), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        return [sys.executable, str(CHILD)], env, FALLBACK_FEATURES

    def _child_env(self) -> dict[str, str]:
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
               "LANG": os.environ.get("LANG", "C.UTF-8"), "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
               "MPLCONFIGDIR": "/tmp", "XDG_CACHE_HOME": "/tmp", "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "RFT_SANDBOX_TIMEOUT_S": str(self.timeout_s), "RFT_SANDBOX_MEM_MB": str(self.mem_mb),
               "RFT_SANDBOX_MAX_OUTPUT": str(4 * self.max_output),
               "RFT_SANDBOX_PRELOAD": ",".join(self.preload)}
        # interpreter discovery only: never PYTHONPATH, which could expose the snapshot or corpus
        for key in ("VIRTUAL_ENV", "PYTHONHOME", "LD_LIBRARY_PATH"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        return env

    def _bwrap(self) -> list[str]:
        argv = [shutil.which("bwrap"), "--unshare-user", "--unshare-net", "--unshare-pid", "--unshare-ipc",
                "--unshare-uts", "--die-with-parent", "--ro-bind", "/", "/", "--tmpfs", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp"]
        masked, interpreter = _masked_dirs()
        for directory in masked:
            argv += ["--tmpfs", directory]
        for directory in interpreter:   # Python must still start: its own directories come back read-only
            argv += ["--ro-bind", directory, directory]
        # the child script's own directory may be masked; bwrap resolves the bind source outside
        argv += ["--ro-bind", str(CHILD), CHILD_IN_SANDBOX, "--chdir", "/tmp", "--clearenv"]
        for key, value in self._child_env().items():
            argv += ["--setenv", key, value]
        return argv + [sys.executable, CHILD_IN_SANDBOX]

    def _render(self, out: bytes, err: bytes, returncode: int) -> str:
        if not out:
            tail = (err or b"").decode("utf-8", "replace").strip()[-500:]
            if returncode and returncode < 0:
                return self._clip(f"Error: compute child terminated by signal {-returncode} (killed: OOM / resource limit).")
            return self._clip(f"Error: compute produced no result (rc={returncode}).{(' stderr: ' + tail) if tail else ''}")
        try:
            data = json.loads(out.decode("utf-8", "replace").strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return "Error: compute returned an unparseable result."
        if data.get("err"):
            before = str(data.get("stdout") or "").strip()
            message = f"Error: {str(data['err'])[:2000]}"
            return self._clip(f"{message}\nOutput before error:\n{before[:1500]}" if before else message)
        parts = []
        if data.get("result") is not None:
            parts.append(f"Result: {data['result']}")
        stdout = str(data.get("stdout") or "").rstrip()
        if stdout:
            parts.append(f"Output:\n{stdout}")
        return self._clip("\n\n".join(parts) if parts else "(code executed successfully, no output)")

    def _clip(self, text: str) -> str:
        return text if len(text) <= self.max_output else text[:self.max_output] + "\n... (output truncated)"


def _communicate(process: subprocess.Popen, code: bytes, timeout_s: float) -> tuple[bytes, bytes, bool]:
    """`(stdout, stderr, timed_out)`. Like `communicate`, but each stream is read only up to
    `READ_LIMIT` bytes (and one more, to tell): a child that writes more is killed, so its output
    never fills the parent's memory."""
    streams: dict[str, bytes] = {}

    def drain(name, stream):
        streams[name] = stream.read(READ_LIMIT + 1)
        if len(streams[name]) > READ_LIMIT:
            process.kill()
        stream.close()

    readers = [threading.Thread(target=drain, args=item, daemon=True)
               for item in (("out", process.stdout), ("err", process.stderr))]
    for reader in readers:
        reader.start()
    try:
        process.stdin.write(code)
        process.stdin.close()
    except (BrokenPipeError, OSError):
        pass   # the child exited before reading all of it; its output says why
    timed_out = False
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()   # --die-with-parent takes the child with bwrap
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for reader in readers:
        reader.join(timeout=5)
    return streams.get("out", b""), streams.get("err", b""), timed_out


def _masked_dirs() -> tuple[list[str], list[str]]:
    """The directories to mask, and the interpreter's directories inside them to bind back."""
    interpreter = {os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix),
                   os.path.dirname(os.path.realpath(sys.executable))}
    masked, inside = [d for d in MASKED if os.path.isdir(d)], set()
    for directory in [*masked, "/tmp"]:   # /tmp is private in the sandbox: an interpreter there comes back too
        real = os.path.realpath(directory)
        inside |= {path for path in interpreter if path == real or path.startswith(real + os.sep)}
    # a directory already inside another bound one needs no bind of its own
    return masked, sorted(p for p in inside if not any(p != q and p.startswith(q + os.sep) for q in inside))
