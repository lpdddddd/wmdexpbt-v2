"""
WM checkpoint manager: saves/loads WM weights + optimizer state + strategy.
Includes aux_value_head state dict for value_imag strategies.
"""

import json
import torch
from pathlib import Path
from typing import Optional


class WMCheckpointManager:
    """
    Saves and loads world model checkpoints.
    Includes: wm_module state, wm_optimizer state, strategy name, iteration.

    Stores aux_value_head state dict alongside WM weights for value_imag strategies.
    """

    def __init__(self, policy_workspace: Path | str):
        self.wm_dir = Path(policy_workspace) / "wm_checkpoints"
        self.wm_dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self.wm_dir / "wm_latest.pt"

    @property
    def latest_path(self) -> Path:
        return self._latest_path

    def save(
        self,
        wm_module,
        wm_optimizer,
        strategy: str,
        iteration: int,
        frames: int,
        aux_value_head_state: Optional[dict] = None,
    ) -> str:
        """
        Save WM checkpoint.
        Returns path to saved checkpoint.
        """
        ckpt = {
            "wm_module_state_dict": wm_module.state_dict(),
            "wm_optimizer_state_dict": wm_optimizer.state_dict(),
            "strategy": strategy,
            "iteration": iteration,
            "frames": frames,
        }
        if hasattr(wm_module, "obs_normalizer") and getattr(wm_module, "obs_normalizer") is not None:
            try:
                ckpt["wm_normalizer_state_dict"] = wm_module.obs_normalizer.state_dict()
            except Exception:
                pass
        if aux_value_head_state is not None:
            ckpt["aux_value_head_state_dict"] = aux_value_head_state

        path = self.wm_dir / f"wm_iter{iteration:08d}.pt"
        torch.save(ckpt, path)

        # Update latest symlink-like path
        torch.save(ckpt, self._latest_path)

        # Write strategy.txt
        (self.wm_dir / "strategy.txt").write_text(strategy)

        return str(path)

    def load_latest(self, wm_module, wm_optimizer,
                    aux_value_head: Optional[torch.nn.Module] = None) -> bool:
        """
        Load from latest checkpoint if exists.
        Returns True if loaded, False if no checkpoint found.
        """
        if not self._latest_path.exists():
            return False

        device = getattr(wm_module, "device", "cpu")
        ckpt = torch.load(self._latest_path, map_location=device)
        wm_module.load_state_dict(ckpt["wm_module_state_dict"], strict=False)
        norm_sd = ckpt.get("wm_normalizer_state_dict")
        if isinstance(norm_sd, dict) and hasattr(wm_module, "obs_normalizer") and getattr(wm_module, "obs_normalizer") is not None:
            try:
                wm_module.obs_normalizer.load_state_dict(norm_sd, strict=False)
            except Exception:
                pass
        if wm_optimizer is not None and "wm_optimizer_state_dict" in ckpt:
            try:
                wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
            except Exception as e:
                print(f"[WM] Failed to load optimizer state: {e}")

        # Restore aux_value_head if present
        if aux_value_head is not None and "aux_value_head_state_dict" in ckpt:
            aux_value_head.load_state_dict(ckpt["aux_value_head_state_dict"], strict=False)

        print(f"[WM] Loaded checkpoint from {self._latest_path} (strategy={ckpt.get('strategy', 'unknown')})")
        return True

    def load_iteration(self, wm_module, wm_optimizer, iteration: int,
                       aux_value_head: Optional[torch.nn.Module] = None) -> bool:
        """Load a specific iteration checkpoint."""
        path = self.wm_dir / f"wm_iter{iteration:08d}.pt"
        if not path.exists():
            return False
        device = getattr(wm_module, "device", "cpu")
        ckpt = torch.load(path, map_location=device)
        wm_module.load_state_dict(ckpt["wm_module_state_dict"], strict=False)
        norm_sd = ckpt.get("wm_normalizer_state_dict")
        if isinstance(norm_sd, dict) and hasattr(wm_module, "obs_normalizer") and getattr(wm_module, "obs_normalizer") is not None:
            try:
                wm_module.obs_normalizer.load_state_dict(norm_sd, strict=False)
            except Exception:
                pass
        if wm_optimizer is not None and "wm_optimizer_state_dict" in ckpt:
            try:
                wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
            except Exception:
                pass
        if aux_value_head is not None and "aux_value_head_state_dict" in ckpt:
            aux_value_head.load_state_dict(ckpt["aux_value_head_state_dict"], strict=False)
        return True

    def get_strategy(self) -> Optional[str]:
        """Read current strategy from strategy.txt."""
        p = self.wm_dir / "strategy.txt"
        if p.exists():
            return p.read_text().strip()
        return None