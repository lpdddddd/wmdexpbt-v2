"""Shared hermetic fixtures for DexPBT unit tests (no simulator / GPU)."""

from __future__ import annotations

import sys
import types


def _stub_rl_games() -> None:
    """Provide a minimal IsaacAlgoObserver so observer imports work offline."""
    if "rl_games.common.algo_observer" in sys.modules and hasattr(
        sys.modules["rl_games.common.algo_observer"], "IsaacAlgoObserver"
    ):
        return
    for key in list(sys.modules):
        if key == "rl_games" or key.startswith("rl_games."):
            del sys.modules[key]
    m = types.ModuleType("rl_games")
    c = types.ModuleType("rl_games.common")
    a = types.ModuleType("rl_games.common.algo_observer")

    class IsaacAlgoObserver:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    a.IsaacAlgoObserver = IsaacAlgoObserver
    sys.modules["rl_games"] = m
    sys.modules["rl_games.common"] = c
    sys.modules["rl_games.common.algo_observer"] = a


_stub_rl_games()
