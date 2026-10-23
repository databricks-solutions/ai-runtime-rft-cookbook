"""Helpers for the adapter tests."""
from __future__ import annotations

from pathlib import Path


def compose_verl(overrides: list[str]):
    """verl's own ppo_trainer config, composed by Hydra with `overrides`."""
    import verl
    from hydra import compose, initialize_config_dir

    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)
