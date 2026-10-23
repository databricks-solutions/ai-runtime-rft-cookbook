"""Task registry: `load_task("math", max_turns=6)` imports `rft_cookbook.tasks.math` and calls its
`task(**budget_overrides)`. A task that needs runtime state (a corpus, a tool service, a judge)
takes `resources`; `load_task(name, resources={...})` passes them through."""
from __future__ import annotations

import importlib
from typing import Any

from rft_cookbook.agent.spec import AgentTask


def load_task(name: str, resources: dict[str, Any] | None = None, **budget_overrides: Any) -> AgentTask:
    module = importlib.import_module(f"rft_cookbook.tasks.{name.replace('-', '_')}")
    task = module.task(resources, **budget_overrides) if resources is not None else module.task(**budget_overrides)
    if task.name != name.replace("-", "_") and task.name != name:
        raise ValueError(f"module rft_cookbook.tasks.{name} built task {task.name!r}")
    return task
