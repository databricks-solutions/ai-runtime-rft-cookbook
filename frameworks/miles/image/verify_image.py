"""Check, inside the built image, that it can run upstream miles' GLM-5.3-Flash path.

    python verify_image.py

The image carries miles' dependencies, not miles: the cookbook installs the pinned miles at job
start (frameworks/miles/rft_miles/miles.lock.json). This checks what that miles needs from them:
the SGLang features it asserts on, the GLM-5.3-Flash model, and the kernels it imports; and what
the cookbook's tasks need (OfficeQA's retrieval, judge credentials and compute sandbox).
"""
from __future__ import annotations

import dataclasses
import importlib
import shutil
import sys


def fields(record) -> set[str]:
    if dataclasses.is_dataclass(record):
        return {f.name for f in dataclasses.fields(record)}
    import msgspec

    return {f.name for f in msgspec.structs.fields(record)}


def main() -> int:
    problems = []
    for name in ("torch", "sglang", "megatron.core", "transformer_engine", "flash_attn", "fla", "tilelang",
                 "deep_gemm", "ray", "transformers", "typer", "huggingface_hub",
                 "sglang.srt.models.glm5_next", "sglang.srt.entrypoints.openai.encoding_dsv41",
                 "bm25s", "databricks.sdk.core", "openai", "sympy", "pylatexenc", "datasets"):
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - report every missing piece at once
            problems.append(f"import {name}: {type(error).__name__}: {error}")
    try:
        from sglang.srt.server_args import ServerArgs

        if "gated_launch_port" not in fields(ServerArgs):
            problems.append("SGLang serves no launch gate (ServerArgs.gated_launch_port): miles waits for it")
    except Exception as error:  # noqa: BLE001
        problems.append(f"sglang ServerArgs: {type(error).__name__}: {error}")
    try:
        import miles  # noqa: F401

        problems.append("the image carries miles; the cookbook installs the pinned miles at job start")
    except ImportError:
        pass
    for tool in ("nvcc", "ninja", "ray", "torchrun"):
        if not shutil.which(tool):
            problems.append(f"{tool} is not on PATH (JIT kernels and the launcher need it)")
    if not shutil.which("bwrap"):
        problems.append("bwrap is not on PATH (the strict compute sandbox needs it)")
    import torch

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
    import sglang

    print(f"sglang {getattr(sglang, '__version__', '?')} at {sglang.__file__}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
