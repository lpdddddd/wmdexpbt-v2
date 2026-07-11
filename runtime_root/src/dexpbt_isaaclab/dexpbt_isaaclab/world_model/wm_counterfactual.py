"""
Counterfactual manager: detects CF triggers and manages file protocol.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


class CounterfactualManager:
    """
    Manages counterfactual block triggering and file protocol.

    Main worker writes cf_block_request.json, CF script reads it,
    runs children, writes cf_block_result.json, main worker reads it.
    """

    def __init__(
        self,
        workspace: Path | str,
        strategies: list[str],
        interval_gens: int = 5,
        env_task: str = "DexPBT-Regrasping-KukaAllegro-Direct-v0",
        num_envs: int = 4096,
        source_num_envs: int = 8192,
        max_parallel_children: int = 2,
        cf_cuda_visible_devices: str = "4,5",
        cf_horizon: int = 16,
        cf_steps: int = 5_000_000,
        # Patch 3 + 4 (rev 2, 2026-05-30): per-block CF child + delta_w
        # configuration. All have safe defaults that match
        # run_counterfactual_block.py constants.
        cf_train_frames: int = 20_000_000,
        cf_reset_epoch_counter: bool = True,
        cf_min_eval_s: float = 30.0,
        cf_delta_mode: str = "learning_improvement",
        cf_low_signal_action: str = "skip",
        cf_low_signal_weight: float = 0.25,
        cf_require_nsucc_gt_zero: bool = True,
        # Patch I (2026-05-31): tighter CF source gates and delta-metric
        # switch. The `_min_nsucc_mean/max/_min_ranking_score` thresholds
        # extend the Patch 3 `nsucc>0` gate so CF blocks only fire on
        # genuinely healthy sources. `cf_delta_metric` selects between
        # the legacy r_meta-based delta and the new nsucc-based delta.
        cf_source_min_nsucc_mean: float = 0.0,
        cf_source_min_nsucc_max: float = 0.0,
        cf_source_min_ranking_score: float = float("-inf"),
        cf_delta_metric: str = "r_meta",
    ):
        self.workspace = Path(workspace)
        self.strategies = strategies
        self.interval_gens = interval_gens
        self.env_task = env_task
        self.num_envs = num_envs
        self.source_num_envs = source_num_envs
        self.max_parallel_children = max_parallel_children
        self.cf_cuda_visible_devices = cf_cuda_visible_devices
        self.cf_horizon = cf_horizon
        self.cf_steps = cf_steps
        # Patch 3 + 4 (rev 2, 2026-05-30): see __init__ docstring.
        self.cf_train_frames = int(cf_train_frames)
        self.cf_reset_epoch_counter = bool(cf_reset_epoch_counter)
        self.cf_min_eval_s = float(cf_min_eval_s)
        self.cf_delta_mode = str(cf_delta_mode)
        self.cf_low_signal_action = str(cf_low_signal_action)
        self.cf_low_signal_weight = float(cf_low_signal_weight)
        self.cf_require_nsucc_gt_zero = bool(cf_require_nsucc_gt_zero)
        # Patch I (2026-05-31): health gates + delta-metric switch.
        self.cf_source_min_nsucc_mean = float(cf_source_min_nsucc_mean)
        self.cf_source_min_nsucc_max = float(cf_source_min_nsucc_max)
        self.cf_source_min_ranking_score = float(cf_source_min_ranking_score)
        if cf_delta_metric not in ("r_meta", "nsucc"):
            raise ValueError(
                f"Unknown cf_delta_metric={cf_delta_metric!r} "
                f"(expected r_meta | nsucc)"
            )
        self.cf_delta_metric = str(cf_delta_metric)

        self._last_trigger_gen = -1
        self._pending_block_id: Optional[int] = None
        # Patch 3 (rev 2): counters surfaced by pbt_observer to TB for
        # quick "is the gate firing?" inspection.
        self.skipped_nsucc_gate_count: int = 0
        # Patch I (2026-05-31): tier-specific counters so TB can tell
        # apart "skipped because <0 nsucc" (Patch 3) from "skipped because
        # source still too weak" (Patch I).
        self.skipped_source_health_gate_count: int = 0

        self.request_path = self.workspace / "cf_block_request.json"
        self.result_path = self.workspace / "cf_block_result.json"

    def check_trigger(
        self,
        pbt_generation: int,
        source_policy_idx: int,
        source_r_meta: float,
        top_checkpoint: str,
        top_params: dict,
        top_tolerance: float,
        top_curriculum: dict | None = None,
        source_env_frames: int | None = None,
        source_nsucc_mean: float | None = None,
        source_nsucc_max: float | None = None,
        source_ranking_score: float | None = None,
        source_meta_frame: int | None = None,
        source_checkpoint_type: str = "live",
    ) -> bool:
        """
        Check if CF block should be triggered.
        Returns True if trigger fired and request was written.

        Patch 3 (rev 2, 2026-05-30): adds an nsucc-gate so we don't burn
        CF compute on a source whose successes are zero (the apparent
        high `source_r_meta` is then purely tolerance progress -- the
        block would inevitably be marked low_signal anyway).
        """
        if pbt_generation - self._last_trigger_gen < self.interval_gens:
            return False

        # Patch 3 nsucc-gate. The caller (`pbt_observer._check_cf_trigger`)
        # has already read curriculum["mean_prev_episode_successes"] from
        # the top worker's meta.json; we re-check here for defence in depth.
        if self.cf_require_nsucc_gt_zero:
            effective_nsucc = source_nsucc_mean
            if effective_nsucc is None:
                effective_nsucc = float(
                    (top_curriculum or {}).get(
                        "mean_prev_episode_successes", 0.0
                    ) or 0.0
                )
            if float(effective_nsucc) <= 0.0:
                self.skipped_nsucc_gate_count += 1
                print(
                    f"[CF] skip trigger: source policy has nsucc_mean="
                    f"{effective_nsucc} <= 0 (gate=cf_require_nsucc_gt_zero)"
                )
                return False

        # Patch I (2026-05-31): tighter source health gates. We only burn
        # CF compute when the source is doing meaningfully better than
        # "barely above zero". Any None scalar passes; the caller is
        # responsible for filling them in from meta.json when available.
        eff_mean = (
            float(source_nsucc_mean)
            if source_nsucc_mean is not None
            else float(
                (top_curriculum or {}).get("mean_prev_episode_successes", 0.0)
                or 0.0
            )
        )
        eff_max = (
            float(source_nsucc_max)
            if source_nsucc_max is not None
            else float(
                (top_curriculum or {}).get("max_prev_episode_successes", 0.0)
                or 0.0
            )
        )
        eff_rs = (
            float(source_ranking_score)
            if source_ranking_score is not None
            else float(source_r_meta)
        )
        if (
            eff_mean < self.cf_source_min_nsucc_mean
            or eff_max < self.cf_source_min_nsucc_max
            or eff_rs < self.cf_source_min_ranking_score
        ):
            self.skipped_source_health_gate_count += 1
            print(
                f"[CF] skip trigger: source health below gate -- "
                f"nsucc_mean={eff_mean:.3f} (min={self.cf_source_min_nsucc_mean}) "
                f"nsucc_max={eff_max:.3f} (min={self.cf_source_min_nsucc_max}) "
                f"ranking_score={eff_rs:.4f} "
                f"(min={self.cf_source_min_ranking_score})"
            )
            return False

        self._last_trigger_gen = pbt_generation
        self._pending_block_id = pbt_generation

        request = {
            "block_id": pbt_generation,
            "source_policy_idx": int(source_policy_idx),
            "source_r_meta": float(source_r_meta),
            "source_checkpoint": top_checkpoint,
            "source_params": top_params,
            "source_tolerance": top_tolerance,
            "source_target_tolerance": float(
                (top_curriculum or {}).get("target_tolerance", top_tolerance)
            ),
            "source_curriculum": dict(top_curriculum or {}),
            "source_env_frames": int(source_env_frames or 0),
            # Patch 3 (rev 2): record source nsucc / ranking_score so the
            # runner / posterior selector can validate "before" snapshots.
            "source_nsucc_mean": (
                None if source_nsucc_mean is None else float(source_nsucc_mean)
            ),
            "source_nsucc_max": (
                None if source_nsucc_max is None else float(source_nsucc_max)
            ),
            "source_ranking_score": (
                None
                if source_ranking_score is None
                else float(source_ranking_score)
            ),
            "source_meta_frame": int(source_meta_frame or source_env_frames or 0),
            "source_checkpoint_type": str(source_checkpoint_type or "live"),
            "strategies": self.strategies,
            "env_task": self.env_task,
            "num_envs": self.num_envs,
            "source_num_envs": self.source_num_envs,
            "cf_num_envs": self.num_envs,
            "cf_horizon": self.cf_horizon,
            "cf_max_parallel_children": self.max_parallel_children,
            "cf_cuda_visible_devices": self.cf_cuda_visible_devices,
            "cf_steps": self.cf_steps,
            # Patch 3 + 4 (rev 2): CF child training/validation budgets and
            # delta_w mode are explicit per-block (writer can override).
            "cf_train_frames": self.cf_train_frames,
            "cf_reset_epoch_counter": self.cf_reset_epoch_counter,
            "cf_min_eval_s": self.cf_min_eval_s,
            "cf_delta_mode": self.cf_delta_mode,
            "cf_low_signal_action": self.cf_low_signal_action,
            "cf_low_signal_weight": self.cf_low_signal_weight,
            # Patch I (2026-05-31): delta-metric switch + health gates
            # surfaced into the request so the runner / analyzer can
            # tell apart blocks fired under different configurations.
            "cf_delta_metric": self.cf_delta_metric,
            "cf_source_min_nsucc_mean": self.cf_source_min_nsucc_mean,
            "cf_source_min_nsucc_max": self.cf_source_min_nsucc_max,
            "cf_source_min_ranking_score": self.cf_source_min_ranking_score,
            "timestamp": time.time(),
        }

        self._write_json_atomic(self.request_path, request)
        print(f"[CF] Triggered block {pbt_generation}, wrote request → {self.request_path}")
        return True

    def peek_unconsumed_result(self, last_consumed_block_id: int = -1) -> dict | None:
        """Read a CF result without consuming it; skip already-seen block ids."""
        if not self.result_path.exists():
            return None
        try:
            result = json.loads(self.result_path.read_text())
        except Exception:
            return None
        block_id = result.get("block_id")
        if block_id is None:
            return None
        if int(block_id) <= int(last_consumed_block_id):
            return None
        for s in self.strategies:
            if s not in result.get("results", {}):
                return None
        return result

    def consume_result_file(self) -> None:
        """Remove the result file after a successful posterior consume."""
        self.result_path.unlink(missing_ok=True)
        self._pending_block_id = None

    def has_result(self) -> bool:
        """Poll for cf_block_result.json with matching block_id."""
        if not self.result_path.exists():
            return False
        try:
            result = json.loads(self.result_path.read_text())
        except Exception:
            return False

        if result.get("block_id") != self._pending_block_id:
            return False

        # Validate all strategies present
        for s in self.strategies:
            if s not in result.get("results", {}):
                return False

        return True

    def get_result(self) -> dict:
        """
        Read and consume the result file.
        Returns full result payload (includes delta_w/results/timestamp).
        """
        if not self.has_result():
            return {}

        result = json.loads(self.result_path.read_text())
        self.result_path.unlink(missing_ok=True)
        self._pending_block_id = None

        return result

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        """Atomic write: write to .tmp, fsync, then rename."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        tmp.rename(path)