"""The core never imports a framework; the installed package works outside the checkout; nothing
workspace-specific or secret is committed."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FRAMEWORKS = {"verl", "miles", "slime", "torch", "vllm", "sglang", "ray", "megatron", "transformer_engine"}
# Lazily imported where they are used, and present in every framework image.
LAZY_ONLY = {"transformers", "openai", "datasets", "huggingface_hub", "pandas", "sympy", "pylatexenc"}


def _imports(path: Path) -> list[tuple[str, bool]]:
    """(top-level module, imported at module level) for every import in a file."""
    tree = ast.parse(path.read_text())
    top = {id(node) for node in tree.body}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name.split(".")[0], id(node) in top) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module.split(".")[0], id(node) in top))
    return found


@pytest.mark.parametrize("path", sorted((REPO / "src").rglob("*.py")), ids=lambda p: str(p.relative_to(REPO)))
def test_core_modules_import_no_framework(path):
    vendored = "_vendor" in path.parts   # imported lazily as a whole by its one caller
    for module, at_top in _imports(path):
        assert module not in FRAMEWORKS, f"{path} imports {module}"
        assert vendored or not (at_top and module in LAZY_ONLY), f"{path} imports {module} at module level"


def test_the_installed_core_imports_outside_the_checkout(tmp_path):
    target = tmp_path / "site"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-build-isolation",
                    "--target", str(target), str(REPO)], check=True, cwd=tmp_path)
    code = ("import sys; import rft_cookbook.tasks.math as m, rft_cookbook.agent.harness, rft_cookbook.checkpoints, "
            "rft_cookbook.evidence.report, rft_cookbook.eval.run; t = m.task(); "
            "assert 'site' in m.__file__, m.__file__; assert 'verl' not in sys.modules; print(t.name)")
    out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(target)!r}); {code}"],
                         capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "math"


# ---- public safety -------------------------------------------------------------------------

PATTERNS = {
    "e-mail address": re.compile(r"\b(?!bugbounty@databricks\.com)[A-Za-z0-9._%+-]+@(?!example\.)[A-Za-z0-9-]+\.(?:com|net|org|io|ai)\b"),
    "workspace host": re.compile(r"\b(?:dbc-[0-9a-f]{8}-[0-9a-f]{4}|adb-\d{6,})\b"
                                 r"|\.cloud\.databricks\.com|\.azuredatabricks\.net|\.gcp\.databricks\.com"),
    "user workspace path": re.compile(r"/Workspace/Users/[A-Za-z0-9]"),
    "volume path": re.compile(r"/Volumes/(?!<|cat/|catalog/)[A-Za-z0-9_]+/"),
    "Databricks token": re.compile(r"\bdapi[0-9a-f]{32}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "run URL": re.compile(r"#job/\d+/run/\d+|/jobs/\d+/runs/\d+"),
}
SCANNED = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh", ".svg", ".jsonl", ".cfg", ".env",
           ".lock", ".in", ".cu"}


def _tracked_files() -> list[Path]:
    skip = {".git", ".venv", ".venv-verl", "__pycache__", ".pytest_cache", "build", "dist"}
    return [p for p in REPO.rglob("*") if p.is_file() and (p.suffix in SCANNED or p.name == "Dockerfile")
            and not any(part in skip or part.endswith(".egg-info") for part in p.relative_to(REPO).parts)
            and p.name not in ("site.yaml", "site.env")]


@pytest.mark.parametrize("path", _tracked_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_nothing_workspace_specific_is_committed(path):
    text = path.read_text(errors="ignore")
    for what, pattern in PATTERNS.items():
        hit = pattern.search(text)
        assert hit is None, f"{path.relative_to(REPO)}: {what}: {hit.group(0)!r}"
