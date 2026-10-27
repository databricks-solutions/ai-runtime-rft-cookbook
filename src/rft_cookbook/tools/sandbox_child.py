# Derived from verl-on-air (Apache License 2.0), usecases/officeqa/sandbox.py at b088a42 (its child
# side); modified for ai-runtime-rft-cookbook: a standalone script configured by RFT_SANDBOX_*
# variables, its output kept to RFT_SANDBOX_MAX_OUTPUT. See THIRD_PARTY_NOTICES.md.
"""The child side of `rft_cookbook.tools.sandbox`: one fresh interpreter per call.

Reads Python from stdin, runs it as ordinary Python (the isolation boundary is the kernel
namespace around this process, not anything in here) and writes one JSON line to stdout:
`{"ok", "stdout", "result", "err"}`. A trailing expression's value is the result. Imports only the
standard library and the configured preload modules, so it runs where nothing else is visible.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys


def main() -> None:
    timeout_s = float(os.environ.get("RFT_SANDBOX_TIMEOUT_S", "15"))
    mem_mb = int(os.environ.get("RFT_SANDBOX_MEM_MB", "0"))
    max_output = int(os.environ.get("RFT_SANDBOX_MAX_OUTPUT", "0"))   # the parent clips to less
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s) + 5, int(timeout_s) + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if mem_mb > 0:
            limit = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001 - limits are best effort inside the child
        pass

    code = sys.stdin.read()
    scope: dict[str, object] = {"__name__": "__compute__", "__builtins__": __builtins__}
    aliases = {"numpy": ("numpy", "np"), "pandas": ("pandas", "pd")}
    for module in filter(None, os.environ.get("RFT_SANDBOX_PRELOAD", "").split(",")):
        try:
            loaded = __import__(module)
            for name in aliases.get(module, (module,)):
                scope[name] = loaded
        except Exception:  # noqa: BLE001
            pass

    result: dict[str, object] = {"ok": False, "stdout": "", "result": None, "err": None}
    buffer = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):  # capture a trailing expression
            last = tree.body.pop()
            tree.body.append(ast.Assign(targets=[ast.Name(id="__result__", ctx=ast.Store())], value=last.value))
            ast.fix_missing_locations(tree)
        compiled = compile(tree, "<compute>", "exec")
    except SyntaxError as error:
        result["err"] = f"SyntaxError: {error}"
        os.write(1, json.dumps(result).encode("utf-8", "replace"))
        return
    except Exception as error:  # noqa: BLE001
        result["err"] = f"{type(error).__name__}: {error}"
        os.write(1, json.dumps(result).encode("utf-8", "replace"))
        return
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compiled, scope)  # noqa: S102 - this process is the sandboxed child
        result["ok"] = True
        value = scope.get("__result__")
        result["result"] = None if value is None else repr(value)[:max_output or None]
    except Exception as error:  # noqa: BLE001
        result["err"] = f"{type(error).__name__}: {error}"
    result["stdout"] = buffer.getvalue()[:max_output or None]
    os.write(1, json.dumps(result).encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
