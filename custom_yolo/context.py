from __future__ import annotations

from copy import deepcopy
from typing import Any

_EXPERIMENT_CONTEXT: dict[str, Any] = {}


def set_experiment_context(context: dict[str, Any] | None) -> None:
    global _EXPERIMENT_CONTEXT
    _EXPERIMENT_CONTEXT = deepcopy(context or {})


def get_experiment_context() -> dict[str, Any]:
    return deepcopy(_EXPERIMENT_CONTEXT)


def clear_experiment_context() -> None:
    set_experiment_context({})
