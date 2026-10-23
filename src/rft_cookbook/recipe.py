"""Recipes: YAML that pins one task to one framework, model and GPU layout.

`extends: agentic/verl` names a parent under `recipes/` (without `.yaml`; a directory means its
`recipe.yaml`). Mappings merge recursively; a child's list or scalar replaces the parent's;
`--set a.b=c` overrides apply last. Unknown top-level keys and `extends` cycles are errors. The
resolved recipe is hashed; the hash goes into every run's plan and qualification record.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

TOP_LEVEL = {"extends", "name", "task", "framework", "model", "data", "agent", "scoring", "compute", "train",
             "strategy", "rollout", "checkpoint", "eval", "parity", "stages", "image", "env", "overrides",
             "description"}


class RecipeError(ValueError):
    pass


def recipes_root(start: Path | None = None) -> Path:
    """`$RFT_RECIPES` (a job's code snapshot: the package itself is installed elsewhere), or the
    repository's `recipes/` above `start`."""
    if start is None and os.environ.get("RFT_RECIPES"):
        return Path(os.environ["RFT_RECIPES"])
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "recipes").is_dir() and (parent / "pyproject.toml").is_file():
            return parent / "recipes"
    raise RecipeError("cannot find the repository's recipes/ directory")


def _path(root: Path, ref: str) -> Path:
    candidate = root / ref
    if candidate.is_dir():
        candidate = candidate / "recipe.yaml"
    elif candidate.suffix != ".yaml":   # appended, not replaced: recipe names hold dots (qwen3.5-2b)
        candidate = candidate.with_name(candidate.name + ".yaml")
    if not candidate.is_file():
        raise RecipeError(f"no recipe {ref!r} under {root}")
    return candidate


def merge(base: Any, child: Any) -> Any:
    if isinstance(base, dict) and isinstance(child, dict):
        merged = dict(base)
        for key, value in child.items():
            merged[key] = merge(base[key], value) if key in base else copy.deepcopy(value)
        return merged
    return copy.deepcopy(child)


def _load(root: Path, ref: str, chain: tuple[str, ...]) -> dict[str, Any]:
    if ref in chain:
        raise RecipeError(f"extends cycle: {' -> '.join(chain + (ref,))}")
    path = _path(root, ref)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise RecipeError(f"{path}: a recipe is a mapping")
    unknown = set(data) - TOP_LEVEL
    if unknown:
        raise RecipeError(f"{path}: unknown keys {sorted(unknown)}")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return merge(_load(root, parent, chain + (ref,)), data)


def set_value(recipe: dict[str, Any], assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise RecipeError(f"override {assignment!r} is not key=value")
    parts = key.split(".")
    if parts[0] not in TOP_LEVEL:
        raise RecipeError(f"override {assignment!r}: unknown top-level key {parts[0]!r}")
    node = recipe
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise RecipeError(f"override {assignment!r}: {part} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load_recipe(ref: str, *, overrides: list[str] = (), root: Path | None = None) -> dict[str, Any]:
    root = root or recipes_root()
    recipe = _load(root, ref, ())
    for assignment in overrides:
        set_value(recipe, assignment)
    recipe.setdefault("name", ref)
    for key in ("task", "framework"):
        if not recipe.get(key):
            raise RecipeError(f"recipe {ref} resolves without {key!r}")
    recipe["resolved_sha256"] = recipe_hash(recipe)
    return recipe


def recipe_hash(recipe: dict[str, Any]) -> str:
    body = {k: v for k, v in recipe.items() if k != "resolved_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


