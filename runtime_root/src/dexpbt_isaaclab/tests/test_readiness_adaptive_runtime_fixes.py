"""Runtime-fix coverage for Profile 23 readiness-adaptive paths.

Exercises real ledger flocking, graduation fail-closed gates, bank cfg restore,
and freshness ranking — not shallow mocks of the call sites.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
    AdaptivePrecisionConfig,
    STAGE_CONSOLIDATION,
    STAGE_EXPLORATION,
    freshness_adjusted_ranking_score,
    select_rescue_source,
)
from dexpbt_isaaclab.pbt.replacement_reservation import (
    PopulationReplacementLedger,
    ReservationHandle,
    ReservationLeaseStatus,
)
from dexpbt_isaaclab.world_model.eir_mpc import _temporary_bank_tolerance_cfg
from dexpbt_isaaclab.world_model.readiness_adaptive import (
    ReadinessAdaptiveConfig,
    ReadinessAdaptiveState,
    _graduation_raw,
    _valid_prediction_frac,
    apply_wm_graduation,
)
from dexpbt_isaaclab.world_model.wm_head_probs import (
    FAILURE_INSUFFICIENT_SAMPLES,
    FAILURE_INVALID_PARTITION,
    FAILURE_PREDICT_EXCEPTION,
    apply_calibration_failure,
    graduation_outcome_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[4]  # .../wmdexpbt
LAUNCHER = (
    REPO_ROOT
    / "results"
    / "launchers"
    / "start_regrasping_cser_dexpbt_readiness_adaptive_smoke.sh"
)
TRAIN_PY = (
    Path(__file__).resolve().parents[1]
    / "dexpbt_isaaclab"
    / "scripts"
    / "train.py"
)

_LINEAGES_ABC = {0: "A", 1: "B", 2: "C"}


def _ra_cfg(**over) -> ReadinessAdaptiveConfig:
    base = dict(
        readiness_adaptive_enable=True,
        readiness_min_frames=100,
        readiness_required_windows=2,
        readiness_enter_threshold=0.50,
        readiness_exit_threshold=0.20,
        readiness_ramp_frames=1000,
        wm_graduation_enable=True,
        wm_graduation_required_passes=2,
        wm_graduation_early_until_frames=500,
        wm_graduation_min_valid_prediction_frac=0.90,
    )
    base.update(over)
    return ReadinessAdaptiveConfig(**base)


def _train_arg_dests() -> set[str]:
    """Parse train.py ``parser.add_argument("--...")`` option strings → dest names."""
    text = TRAIN_PY.read_text(encoding="utf-8")
    dests: set[str] = set()
    for m in re.finditer(
        r'parser\.add_argument\(\s*"--([^"]+)"',
        text,
    ):
        opt = m.group(1)
        # BooleanOptionalAction registers both --foo and --no-foo → dest=foo
        if opt.startswith("no-"):
            opt = opt[3:]
        dests.add(opt.replace("-", "_"))
    return dests


def _extra_flags_from_launcher(text: str) -> list[str]:
    """Collect ``--*`` tokens after the first ``--extra`` in the launcher script."""
    # Collapse line continuations so ``--extra`` and following flags are contiguous.
    flat = re.sub(r"\\\s*\n", " ", text)
    # Find the train ``--extra`` block (not comments).
    idx = flat.find("--extra")
    if idx < 0:
        return []
    tail = flat[idx + len("--extra") :]
    # Stop at shell redirect / background.
    tail = re.split(r">>|<<|&\s*$", tail, maxsplit=1)[0]
    flags: list[str] = []
    for tok in re.findall(r"(--[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*)", tail):
        if tok == "--extra":
            continue
        flags.append(tok)
    return flags


def _flag_to_dest(flag: str) -> str:
    name = flag[2:] if flag.startswith("--") else flag
    if name.startswith("no-"):
        name = name[3:]
    return name.replace("-", "_")


def _is_readiness_related(dest: str) -> bool:
    return (
        dest.startswith("readiness_")
        or dest.startswith("wm_graduation_")
        or dest.startswith("adaptive_")
    )


def _reserve_kwargs(
    *,
    target_idx: int,
    budget: int,
    min_distinct: int,
    stage: str = STAGE_EXPLORATION,
    source_lineage: str = "A",
    source_idx: int = 0,
) -> dict:
    return dict(
        tick_id=1,
        stage=stage,
        budget=budget,
        min_distinct_lineages=min_distinct,
        population_lineages=dict(_LINEAGES_ABC),
        num_policies=3,
        target_idx=target_idx,
        source_idx=source_idx,
        source_lineage=source_lineage,
        owner_policy=target_idx,
    )


def _mp_try_reserve(workspace: str, target_idx: int, budget: int, min_distinct: int, q):
    ledger = PopulationReplacementLedger(workspace)
    res = ledger.try_reserve_ordinary(**_reserve_kwargs(
        target_idx=target_idx, budget=budget, min_distinct=min_distinct
    ))
    rid = res.handle.reservation_id if res.handle is not None else ""
    q.put((bool(res.ok), str(res.reason), str(rid)))


class TestSmokeLauncherCliParse(unittest.TestCase):
    def test_extra_readiness_flags_match_train_argparse(self):
        self.assertTrue(LAUNCHER.is_file(), f"missing launcher: {LAUNCHER}")
        self.assertTrue(TRAIN_PY.is_file(), f"missing train.py: {TRAIN_PY}")
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("readiness_adaptivement", text)

        flags = _extra_flags_from_launcher(text)
        self.assertIn("--readiness_min_frames", flags)
        self.assertIn("--readiness_ramp_frames", flags)

        dests = _train_arg_dests()
        readiness_flags = [
            f for f in flags if _is_readiness_related(_flag_to_dest(f))
        ]
        self.assertGreaterEqual(len(readiness_flags), 2)
        for flag in readiness_flags:
            dest = _flag_to_dest(flag)
            self.assertIn(
                dest,
                dests,
                f"launcher flag {flag} (dest={dest}) missing from train.py argparse",
            )


class TestCrossProcessBudget(unittest.TestCase):
    def test_budget_one_exactly_one_reservation_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mp.get_context("spawn")
            q: mp.Queue = ctx.Queue()
            procs = [
                ctx.Process(
                    target=_mp_try_reserve,
                    args=(tmp, tgt, 1, 1, q),
                )
                for tgt in (1, 2)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=30)
                self.assertEqual(p.exitcode, 0, "worker process failed")

            results = [q.get(timeout=5) for _ in range(2)]
            oks = [r[0] for r in results]
            self.assertEqual(sum(1 for ok in oks if ok), 1, results)


class TestConcurrentLineage(unittest.TestCase):
    def _concurrent_reserve(
        self,
        workspace: str,
        *,
        budget: int,
        min_distinct: int,
        stage: str,
        targets: tuple[int, int] = (1, 2),
    ) -> list[bool]:
        ledger = PopulationReplacementLedger(workspace)
        barrier = threading.Barrier(2)
        out: list[bool | None] = [None, None]

        def _worker(slot: int, target: int) -> None:
            barrier.wait(timeout=10)
            res = ledger.try_reserve_ordinary(
                **_reserve_kwargs(
                    target_idx=target,
                    budget=budget,
                    min_distinct=min_distinct,
                    stage=stage,
                )
            )
            out[slot] = bool(res.ok)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(_worker, i, t) for i, t in enumerate(targets)
            ]
            for f in futs:
                f.result(timeout=30)
        return [bool(x) for x in out]

    def test_exploration_min_distinct_3_not_both_succeed(self):
        # 0:A,1:B,2:C; copying A onto either target alone drops to 2 lineages.
        with tempfile.TemporaryDirectory() as tmp:
            oks = self._concurrent_reserve(
                tmp,
                budget=2,
                min_distinct=3,
                stage=STAGE_EXPLORATION,
            )
            self.assertLessEqual(sum(1 for ok in oks if ok), 1)
            # With exactly 3 lineages, a single A→target copy also violates min=3.
            self.assertEqual(sum(1 for ok in oks if ok), 0)

    def test_consolidation_min_distinct_2_at_most_one(self):
        # budget=2 would allow both on budget alone; diversity must cap at one
        # (first: A,A,C → 2 ok; second: A,A,A → 1 blocked).
        with tempfile.TemporaryDirectory() as tmp:
            oks = self._concurrent_reserve(
                tmp,
                budget=2,
                min_distinct=2,
                stage=STAGE_CONSOLIDATION,
            )
            self.assertEqual(sum(1 for ok in oks if ok), 1, oks)

    def test_consolidation_budget_1_at_most_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            oks = self._concurrent_reserve(
                tmp,
                budget=1,
                min_distinct=1,  # diversity off → budget is the binding constraint
                stage=STAGE_CONSOLIDATION,
            )
            self.assertEqual(sum(1 for ok in oks if ok), 1, oks)


class TestReservationReleaseTimeout(unittest.TestCase):
    def test_release_frees_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp, reservation_timeout_s=300.0)
            r1 = ledger.try_reserve_ordinary(**_reserve_kwargs(target_idx=1, budget=1, min_distinct=1))
            self.assertTrue(r1.ok, r1.reason)
            self.assertIsNotNone(r1.handle)
            r2 = ledger.try_reserve_ordinary(**_reserve_kwargs(target_idx=2, budget=1, min_distinct=1))
            self.assertFalse(r2.ok)
            self.assertEqual(r2.reason, "replacement_budget_exhausted")
            fr = ledger.release(r1.handle)
            self.assertTrue(fr.ok, fr.reason)
            r3 = ledger.try_reserve_ordinary(**_reserve_kwargs(target_idx=2, budget=1, min_distinct=1))
            self.assertTrue(r3.ok, r3.reason)

    def test_expired_reservation_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp, reservation_timeout_s=0.15)
            r1 = ledger.try_reserve_ordinary(**_reserve_kwargs(target_idx=1, budget=1, min_distinct=1))
            self.assertTrue(r1.ok, r1.reason)
            time.sleep(0.25)
            # New try_reserve expires stale entries before budget check.
            r2 = ledger.try_reserve_ordinary(**_reserve_kwargs(target_idx=2, budget=1, min_distinct=1))
            self.assertTrue(r2.ok, r2.reason)
            self.assertNotEqual(
                r1.handle.reservation_id, r2.handle.reservation_id
            )


class TestLedgerHandleApi(unittest.TestCase):
    def test_same_tick_different_stage_different_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            r_exp = ledger.try_reserve_ordinary(
                **_reserve_kwargs(
                    target_idx=1, budget=1, min_distinct=1, stage=STAGE_EXPLORATION
                )
            )
            r_con = ledger.try_reserve_ordinary(
                **_reserve_kwargs(
                    target_idx=1, budget=1, min_distinct=1, stage=STAGE_CONSOLIDATION
                )
            )
            self.assertTrue(r_exp.ok, r_exp.reason)
            self.assertTrue(r_con.ok, r_con.reason)
            p_exp = Path(tmp) / "adaptive_replacement" / f"tick_1_{STAGE_EXPLORATION}.json"
            p_con = Path(tmp) / "adaptive_replacement" / f"tick_1_{STAGE_CONSOLIDATION}.json"
            self.assertTrue(p_exp.is_file())
            self.assertTrue(p_con.is_file())
            self.assertNotEqual(p_exp, p_con)

    def test_budget_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            r1 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=2, min_distinct=1)
            )
            self.assertTrue(r1.ok, r1.reason)
            r2 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=2, budget=1, min_distinct=1)
            )
            self.assertFalse(r2.ok)
            self.assertEqual(r2.reason, "ledger_budget_mismatch")

    def test_corrupt_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            path = Path(tmp) / "adaptive_replacement" / f"tick_1_{STAGE_EXPLORATION}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json")
            r = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=1, min_distinct=1)
            )
            self.assertFalse(r.ok)
            self.assertEqual(r.reason, "ledger_corrupt")

    def test_wrong_stage_commit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            r1 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(
                    target_idx=1, budget=1, min_distinct=1, stage=STAGE_EXPLORATION
                )
            )
            self.assertTrue(r1.ok, r1.reason)
            bad = ReservationHandle(
                tick_id=r1.handle.tick_id,
                stage=STAGE_CONSOLIDATION,
                reservation_id=r1.handle.reservation_id,
                owner_policy=r1.handle.owner_policy,
                owner_pid=r1.handle.owner_pid,
                owner_token=r1.handle.owner_token,
                target=r1.handle.target,
                source=r1.handle.source,
                source_lineage=r1.handle.source_lineage,
                lease_timeout_s=r1.handle.lease_timeout_s,
            )
            fr = ledger.commit(bad)
            self.assertFalse(fr.ok)
            self.assertIn(fr.reason, ("ledger_missing", "ledger_stage_mismatch"))

    def test_heartbeat_prevents_expire_under_short_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(
                tmp, reservation_timeout_s=0.2, heartbeat_interval_s=0.05
            )
            r1 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=1, min_distinct=1)
            )
            self.assertTrue(r1.ok, r1.reason)
            with ledger.reservation_lease(r1.handle, heartbeat_interval_s=0.05):
                time.sleep(0.45)
            # Still reserved (not expired) → budget still held.
            r2 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=2, budget=1, min_distinct=1)
            )
            self.assertFalse(r2.ok)
            self.assertEqual(r2.reason, "replacement_budget_exhausted")
            fr = ledger.commit(r1.handle)
            self.assertTrue(fr.ok, fr.reason)

    def test_non_owner_renew_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            r1 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=1, min_distinct=1)
            )
            self.assertTrue(r1.ok, r1.reason)
            bad = ReservationHandle(
                tick_id=r1.handle.tick_id,
                stage=r1.handle.stage,
                reservation_id=r1.handle.reservation_id,
                owner_policy=r1.handle.owner_policy,
                owner_pid=r1.handle.owner_pid,
                owner_token="not-the-owner-token",
                target=r1.handle.target,
                source=r1.handle.source,
                source_lineage=r1.handle.source_lineage,
                lease_timeout_s=r1.handle.lease_timeout_s,
            )
            fr = ledger.renew(bad)
            self.assertFalse(fr.ok)
            self.assertEqual(fr.reason, "owner_mismatch")

    def test_reconcile_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp, reservation_timeout_s=0.1)
            r1 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=1, min_distinct=1)
            )
            self.assertTrue(r1.ok, r1.reason)
            time.sleep(0.2)
            # Expire via another reserve attempt on same file.
            _ = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=2, budget=1, min_distinct=1)
            )
            fr1 = ledger.reconcile_commit(r1.handle)
            self.assertTrue(fr1.ok, fr1.reason)
            fr2 = ledger.reconcile_commit(r1.handle)
            self.assertTrue(fr2.ok, fr2.reason)
            self.assertEqual(fr2.reason, "already_committed")

    def test_late_rescue_bypass_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            tele = ledger.record_bypass(
                tick_id=1,
                stage=STAGE_EXPLORATION,
                budget=1,
                target_idx=2,
                reason="late_rescue",
            )
            self.assertGreaterEqual(
                float(tele.get("adaptive_consolidation/replacement_budget_bypassed", 0.0)),
                1.0,
            )
            path = Path(tmp) / "adaptive_replacement" / f"tick_1_{STAGE_EXPLORATION}.json"
            data = __import__("json").loads(path.read_text())
            reasons = [str(x.get("reason")) for x in data.get("bypassed", [])]
            self.assertIn("late_rescue", reasons)


class TestCalibrationFailureClassification(unittest.TestCase):
    def test_graduation_outcome_mode_insufficient_vs_structural(self):
        self.assertEqual(
            graduation_outcome_mode(
                eval_valid=False,
                failure_reason_id=FAILURE_INSUFFICIENT_SAMPLES,
                calibration_pass=False,
            ),
            "insufficient",
        )
        self.assertEqual(
            graduation_outcome_mode(
                eval_valid=False,
                failure_reason_id=FAILURE_PREDICT_EXCEPTION,
                calibration_pass=False,
            ),
            "structural",
        )
        self.assertEqual(
            graduation_outcome_mode(
                eval_valid=False,
                failure_reason_id=FAILURE_INVALID_PARTITION,
                calibration_pass=False,
            ),
            "insufficient",
        )

    def test_invalid_partition_apply_calibration_failure_insufficient(self):
        from dexpbt_isaaclab.world_model.wm_calibration_evaluator import (
            WMCalibrationConfig,
            WMCalibrationState,
        )

        cfg = WMCalibrationConfig(enabled=True, ema_beta=0.5)
        st = WMCalibrationState(alpha_wm=0.7, ema_raw=0.7)
        st = apply_calibration_failure(
            st,
            cfg=cfg,
            frames=10,
            metrics={},
            failure_reason_id=FAILURE_INVALID_PARTITION,
        )
        self.assertEqual(st.last_metrics["wm_cal/outcome_mode"], "insufficient")
        self.assertEqual(
            st.last_metrics["wm_cal/failure_reason_id"], FAILURE_INVALID_PARTITION
        )
        self.assertLess(st.ema_raw, 0.7)
        self.assertFalse(st.calibration_pass)

    def test_apply_wm_graduation_insufficient_beta_down_structural_zeros(self):
        cfg = _ra_cfg(wm_graduation_beta_down=0.5)
        st = ReadinessAdaptiveState(
            wm_pass_streak=5, wm_graduated=True, wm_ema_raw=0.8, last_alpha_effective=0.4
        )
        st_ins, alpha_ins = apply_wm_graduation(
            st,
            frames=100,
            metrics={},
            mode="insufficient",
            cfg=cfg,
            r_bank=1.0,
        )
        self.assertAlmostEqual(st_ins.wm_ema_raw, 0.4)
        self.assertEqual(st_ins.wm_pass_streak, 0)
        self.assertEqual(alpha_ins, 0.0)
        self.assertFalse(st_ins.wm_graduated)

        st2 = ReadinessAdaptiveState(
            wm_pass_streak=5, wm_graduated=True, wm_ema_raw=0.8, last_alpha_effective=0.4
        )
        st_str, alpha_str = apply_wm_graduation(
            st2,
            frames=100,
            metrics={},
            mode="structural",
            cfg=cfg,
            r_bank=1.0,
        )
        self.assertEqual(st_str.wm_ema_raw, 0.0)
        self.assertEqual(st_str.wm_pass_streak, 0)
        self.assertEqual(alpha_str, 0.0)
        self.assertFalse(st_str.wm_graduated)

    def test_nan_raw_gate_no_streak(self):
        cfg = _ra_cfg()
        st = ReadinessAdaptiveState()
        st, _ = apply_wm_graduation(
            st,
            frames=100,
            metrics={
                "wm_cal/eval_valid": 1.0,
                "wm_cal/calibration_pass": 1.0,
                "wm_cal/valid_prediction_frac": 1.0,
                "wm_cal/raw_gate": float("nan"),
            },
            mode="valid",
            cfg=cfg,
            r_bank=1.0,
        )
        self.assertEqual(st.wm_pass_streak, 0)
        raw, ok = _graduation_raw({"wm_cal/raw_gate": float("nan")})
        self.assertFalse(ok)
        self.assertEqual(raw, 0.0)
        # calibration_pass alone is not enough for a valid raw.
        raw2, ok2 = _graduation_raw({"wm_cal/calibration_pass": 1.0})
        self.assertFalse(ok2)

    def test_single_gate_nan_invalid_raw(self):
        raw, ok = _graduation_raw(
            {
                "wm_cal/gate_recovery_auc": 1.0,
                "wm_cal/gate_recovery_ece": float("nan"),
                "wm_cal/gate_combined_rank": 1.0,
            }
        )
        self.assertFalse(ok)
        self.assertEqual(raw, 0.0)

    def test_raw_gate_out_of_range_invalid(self):
        for bad in (-0.1, 1.1, float("inf"), float("-inf")):
            raw, ok = _graduation_raw({"wm_cal/raw_gate": bad})
            self.assertFalse(ok, bad)
            self.assertEqual(raw, 0.0)

    def test_r_bank_illegal_rejects_pass(self):
        cfg = _ra_cfg(wm_graduation_required_passes=1)
        metrics = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/valid_prediction_frac": 1.0,
            "wm_cal/raw_gate": 1.0,
        }
        for r_bank in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1, -1.0, 2.0):
            st = ReadinessAdaptiveState(wm_ema_raw=0.5)
            st, alpha = apply_wm_graduation(
                st,
                frames=100,
                metrics=metrics,
                mode="valid",
                cfg=cfg,
                r_bank=r_bank,
            )
            self.assertEqual(st.wm_pass_streak, 0, r_bank)
            self.assertFalse(st.wm_graduated, r_bank)
            self.assertEqual(alpha, 0.0, r_bank)

    def test_r_bank_boundaries_legal(self):
        cfg = _ra_cfg(wm_graduation_required_passes=2)
        metrics = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/valid_prediction_frac": 1.0,
            "wm_cal/raw_gate": 1.0,
        }
        for r_bank in (0.0, 1.0):
            st = ReadinessAdaptiveState(wm_ema_raw=0.0)
            st, _ = apply_wm_graduation(
                st,
                frames=100,
                metrics=metrics,
                mode="valid",
                cfg=cfg,
                r_bank=r_bank,
            )
            self.assertEqual(st.wm_pass_streak, 1, r_bank)
            self.assertFalse(st.wm_graduated, r_bank)

    def test_illegal_ema_no_streak(self):
        cfg = _ra_cfg(wm_graduation_required_passes=1)
        metrics = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/valid_prediction_frac": 1.0,
            "wm_cal/raw_gate": 1.0,
        }
        for ema in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1):
            st = ReadinessAdaptiveState(wm_ema_raw=ema)
            st, alpha = apply_wm_graduation(
                st,
                frames=100,
                metrics=metrics,
                mode="valid",
                cfg=cfg,
                r_bank=1.0,
            )
            self.assertEqual(st.wm_pass_streak, 0, ema)
            self.assertFalse(st.wm_graduated, ema)
            self.assertEqual(st.wm_ema_raw, 0.0, ema)
            self.assertEqual(alpha, 0.0, ema)

    def test_two_legal_passes_graduate(self):
        cfg = _ra_cfg(wm_graduation_required_passes=2)
        metrics = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/valid_prediction_frac": 1.0,
            "wm_cal/raw_gate": 1.0,
        }
        st = ReadinessAdaptiveState(wm_ema_raw=0.0)
        st, a1 = apply_wm_graduation(
            st, frames=100, metrics=metrics, mode="valid", cfg=cfg, r_bank=1.0
        )
        self.assertEqual(st.wm_pass_streak, 1)
        self.assertFalse(st.wm_graduated)
        self.assertEqual(a1, 0.0)
        st, a2 = apply_wm_graduation(
            st, frames=100, metrics=metrics, mode="valid", cfg=cfg, r_bank=1.0
        )
        self.assertEqual(st.wm_pass_streak, 2)
        self.assertTrue(st.wm_graduated)
        self.assertGreater(a2, 0.0)


class TestValidPredictionFracFailClosed(unittest.TestCase):
    def test_missing_and_nan_are_zero(self):
        self.assertEqual(_valid_prediction_frac({}), 0.0)
        self.assertEqual(
            _valid_prediction_frac({"wm_cal/valid_prediction_frac": float("nan")}),
            0.0,
        )
        self.assertEqual(
            _valid_prediction_frac({"wm_cal/valid_prediction_frac": float("inf")}),
            0.0,
        )

    def test_threshold_089_fails_090_passes_with_cal_pass(self):
        cfg = _ra_cfg()
        base = {
            "wm_cal/eval_valid": 1.0,
            "wm_cal/calibration_pass": 1.0,
            "wm_cal/raw_gate": 1.0,
        }
        st = ReadinessAdaptiveState()
        st, _ = apply_wm_graduation(
            st,
            frames=100,
            metrics={**base, "wm_cal/valid_prediction_frac": 0.89},
            mode="valid",
            cfg=cfg,
            r_bank=1.0,
        )
        self.assertEqual(st.wm_pass_streak, 0)

        st2 = ReadinessAdaptiveState()
        st2, _ = apply_wm_graduation(
            st2,
            frames=100,
            metrics={**base, "wm_cal/valid_prediction_frac": 0.90},
            mode="valid",
            cfg=cfg,
            r_bank=1.0,
        )
        self.assertEqual(st2.wm_pass_streak, 1)

        # eval_valid without calibration_pass must not advance the streak.
        st3 = ReadinessAdaptiveState()
        st3, _ = apply_wm_graduation(
            st3,
            frames=100,
            metrics={
                "wm_cal/eval_valid": 1.0,
                "wm_cal/calibration_pass": 0.0,
                "wm_cal/valid_prediction_frac": 0.95,
                "wm_cal/raw_gate": 1.0,
            },
            mode="valid",
            cfg=cfg,
            r_bank=1.0,
        )
        self.assertEqual(st3.wm_pass_streak, 0)


class TestBankCfgRestoreOnException(unittest.TestCase):
    def test_temporary_bank_tolerance_restored_after_raise(self):
        cfg = SimpleNamespace(tolerance_max_log_gap=0.30, tolerance_log_soft_scale=0.15)
        bank = SimpleNamespace(cfg=cfg)
        with self.assertRaises(RuntimeError):
            with _temporary_bank_tolerance_cfg(bank, max_log_gap=0.20, soft_scale=0.10):
                self.assertAlmostEqual(cfg.tolerance_max_log_gap, 0.20)
                self.assertAlmostEqual(cfg.tolerance_log_soft_scale, 0.10)
                raise RuntimeError("boom")
        self.assertAlmostEqual(cfg.tolerance_max_log_gap, 0.30)
        self.assertAlmostEqual(cfg.tolerance_log_soft_scale, 0.15)


class TestFreshnessNegativeScore(unittest.TestCase):
    def _cand(self, idx: int, ranking_score: float, age: int) -> dict:
        return {
            "idx": idx,
            "nsucc_mean": 12.0 + idx,
            "retention": 0.95,
            "tolerance": 0.05,
            "ranking_score": ranking_score,
            "has_control": True,
            "collapsed": False,
            "retention_valid": True,
            "source_age_frames": age,
        }

    def test_exp_decay_older_loses_on_equal_negative_scores(self):
        cfg = AdaptivePrecisionConfig(
            enabled=True,
            rescue_enable=True,
            rescue_source_freshness_mode="exp_decay",
            rescue_source_freshness_tau_frames=80_000_000,
            rescue_source_hard_max_age_frames=160_000_000,
            rescue_source_max_staleness_frames=160_000_000,
            rescue_min_source_advantage_sigma=0.0,
            rescue_source_top_fraction=1.0,
        )
        ok_f, s_fresh, _ = freshness_adjusted_ranking_score(-2.0, 0, cfg)
        ok_o, s_old, _ = freshness_adjusted_ranking_score(-2.0, 80_000_000, cfg)
        self.assertTrue(ok_f and ok_o)
        self.assertGreater(s_fresh, s_old)

        decision = select_rescue_source(
            target_tolerance=0.05,
            target_nsucc_mean=5.0,
            candidates=[
                self._cand(0, -2.0, age=80_000_000),
                self._cand(1, -2.0, age=0),
            ],
            population_nsucc_means=[12.0, 13.0, 5.0],
            cfg=cfg,
            sorted_population=[1, 0, 2],
        )
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.source_idx, 1)  # fresher wins

    def test_hard_max_preserves_equal_negative_order(self):
        cfg = AdaptivePrecisionConfig(
            enabled=True,
            rescue_enable=True,
            rescue_source_freshness_mode="hard_max",
            rescue_source_max_staleness_frames=40_000_000,
            rescue_min_source_advantage_sigma=0.0,
            rescue_source_top_fraction=1.0,
        )
        ok_a, s_a, _ = freshness_adjusted_ranking_score(-2.0, 0, cfg)
        ok_b, s_b, _ = freshness_adjusted_ranking_score(-2.0, 1_000_000, cfg)
        self.assertTrue(ok_a and ok_b)
        self.assertEqual(s_a, s_b)
        self.assertEqual(s_a, -2.0)

        # First candidate in iteration order with equal adjusted scores wins.
        decision = select_rescue_source(
            target_tolerance=0.05,
            target_nsucc_mean=5.0,
            candidates=[
                self._cand(0, -2.0, age=0),
                self._cand(1, -2.0, age=1_000_000),
            ],
            population_nsucc_means=[12.0, 13.0, 5.0],
            cfg=cfg,
            sorted_population=[0, 1, 2],
        )
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.source_idx, 0)


class TestPeerLineageResolve(unittest.TestCase):
    def _cfg(self):
        return AdaptivePrecisionConfig(
            enabled=True,
            rescue_source_max_staleness_frames=50,
            rescue_source_freshness_mode="hard_max",
            rescue_require_lineage_match_with_checkpoint=True,
        )

    def _side(self, *, frame: int, lin: str = "L1", schema: int = 2):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            ADAPTIVE_SIDECAR_SCHEMA_VERSION,
        )

        return {
            "schema_version": int(schema if schema is not None else ADAPTIVE_SIDECAR_SCHEMA_VERSION),
            "adaptive_lineage_id": lin,
            "last_update_lineage_id": lin,
            "last_update_frame": int(frame),
            "last_checkpoint_frame": int(frame),
        }

    def test_population_lineage_preferred(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_peer_lineage_for_reservation,
        )

        lin, reason = resolve_peer_lineage_for_reservation(
            population_state={"lineage_id": "POP", "frame": 10},
            sidecar=self._side(frame=10, lin="SIDE"),
            interval_steps=1,
            cfg=self._cfg(),
        )
        self.assertEqual(lin, "POP")
        self.assertEqual(reason, "lineage_population_ok")

    def test_peer_frame_not_target_frame(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_peer_lineage_for_reservation,
        )

        # Sidecar stale vs target(1000) but fresh vs peer(20).
        side = self._side(frame=15, lin="PEER")
        lin, reason = resolve_peer_lineage_for_reservation(
            population_state={"frame": 20},  # peer frame
            sidecar=side,
            interval_steps=1,
            cfg=self._cfg(),
        )
        self.assertEqual(lin, "PEER")
        self.assertEqual(reason, "lineage_sidecar_ok")

    def test_sidecar_future_rejected(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_peer_lineage_for_reservation,
        )

        lin, reason = resolve_peer_lineage_for_reservation(
            population_state={"frame": 10},
            sidecar=self._side(frame=50),
            interval_steps=1,
            cfg=self._cfg(),
        )
        self.assertEqual(lin, "")
        self.assertEqual(reason, "lineage_sidecar_future")

    def test_peer_frame_missing(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_peer_lineage_for_reservation,
        )

        lin, reason = resolve_peer_lineage_for_reservation(
            population_state={},
            sidecar=self._side(frame=10),
            interval_steps=1,
            cfg=self._cfg(),
        )
        self.assertEqual(lin, "")
        self.assertEqual(reason, "lineage_peer_frame_missing")

    def test_lineage_mismatch(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_peer_lineage_for_reservation,
        )

        side = self._side(frame=10, lin="A")
        side["last_update_lineage_id"] = "B"
        lin, reason = resolve_peer_lineage_for_reservation(
            population_state={"frame": 20},
            sidecar=side,
            interval_steps=1,
            cfg=self._cfg(),
        )
        self.assertEqual(lin, "")
        self.assertEqual(reason, "lineage_sidecar_mismatch")

    def test_stale_sidecar(self):
        from dexpbt_isaaclab.pbt.adaptive_precision_consolidation import (
            resolve_peer_lineage_for_reservation,
        )

        lin, reason = resolve_peer_lineage_for_reservation(
            population_state={"frame": 200},
            sidecar=self._side(frame=10),
            interval_steps=1,
            cfg=self._cfg(),
        )
        self.assertEqual(lin, "")
        self.assertEqual(reason, "lineage_sidecar_stale")


class TestReservationLeaseExceptionSafety(unittest.TestCase):
    def _reserve(self, tmp, **over):
        ledger = PopulationReplacementLedger(
            tmp, reservation_timeout_s=0.5, heartbeat_interval_s=0.05
        )
        r = ledger.try_reserve_ordinary(**_reserve_kwargs(target_idx=1, budget=1, min_distinct=1, **over))
        self.assertTrue(r.ok, r.reason)
        return ledger, r.handle

    def test_body_valueerror_preserved_over_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, handle = self._reserve(tmp)

            def boom():
                raise ValueError("restart-body")

            ok, status, fr, exc = ledger.run_restart_with_reservation(handle, boom)
            self.assertFalse(ok)
            self.assertIsInstance(exc, ValueError)
            self.assertEqual(str(exc), "restart-body")
            self.assertTrue(fr.ok)
            self.assertEqual(fr.reason, "released")

    def test_body_oserror_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, handle = self._reserve(tmp)

            def boom():
                raise OSError("disk")

            ok, status, fr, exc = ledger.run_restart_with_reservation(handle, boom)
            self.assertFalse(ok)
            self.assertIsInstance(exc, OSError)
            self.assertEqual(fr.reason, "released")

    def test_false_restart_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, handle = self._reserve(tmp)
            ok, status, fr, exc = ledger.run_restart_with_reservation(
                handle, lambda: False
            )
            self.assertFalse(ok)
            self.assertIsNone(exc)
            self.assertEqual(fr.reason, "released")
            self.assertTrue(status.thread_stopped)

    def test_success_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, handle = self._reserve(tmp)
            ok, status, fr, exc = ledger.run_restart_with_reservation(
                handle, lambda: True
            )
            self.assertTrue(ok)
            self.assertIsNone(exc)
            self.assertEqual(fr.reason, "committed")
            self.assertTrue(status.heartbeat_ok)

    def test_reconcile_budget_overrun_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp, reservation_timeout_s=0.05)
            # Fill budget via commit of first reservation.
            r1 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=1, min_distinct=1)
            )
            self.assertTrue(r1.ok)
            self.assertTrue(ledger.commit(r1.handle).ok)
            # Second reservation expires then reconcile beyond budget.
            r2 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=2, budget=1, min_distinct=1)
            )
            # Budget exhausted — force reconstruct via reconcile of a forged expired handle
            # by manually creating a reserved-then-expired slot is hard; instead
            # reconcile a handle whose reservation was never on the ledger after
            # reconstructing from handle with budget already full.
            fake = ReservationHandle(
                tick_id=1,
                stage=STAGE_EXPLORATION,
                reservation_id="deadbeef",
                owner_policy=0,
                owner_pid=os.getpid(),
                owner_token="tok",
                target=2,
                source=0,
                source_lineage="A",
                lease_timeout_s=0.05,
            )
            # Need matching owner fields in reconstructed reservation — reconcile
            # reconstructs from handle when missing.
            fr1 = ledger.reconcile_commit(fake)
            self.assertTrue(fr1.ok)
            self.assertEqual(fr1.reason, "reconciled_budget_overrun")
            self.assertEqual(
                fr1.telemetry.get(
                    "adaptive_consolidation/replacement_budget_overrun_reconciled"
                ),
                1.0,
            )
            fr2 = ledger.reconcile_commit(fake)
            self.assertEqual(fr2.reason, "already_committed")
            self.assertEqual(
                fr2.telemetry.get(
                    "adaptive_consolidation/replacement_budget_overrun_reconciled"
                ),
                1.0,
            )

    def test_lease_status_no_raise_on_body_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, handle = self._reserve(tmp)
            status = ReservationLeaseStatus()
            with self.assertRaises(RuntimeError):
                with ledger.reservation_lease(handle, status=status):
                    raise RuntimeError("body")
            # Context exited without replacing body error; status may note heartbeat.
            self.assertTrue(status.thread_stopped)


class TestObserverRestartOrchestration(unittest.TestCase):
    def test_observer_helper_releases_on_oserror(self):
        # Import method without importing full observer module (rl_games dep).
        import importlib.util

        # Exercise ledger orchestration path used by observer helper.
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PopulationReplacementLedger(tmp)
            r = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=1, budget=1, min_distinct=1)
            )
            self.assertTrue(r.ok)

            def boom():
                raise OSError("boom")

            ok, status, fr, exc = ledger.run_restart_with_reservation(r.handle, boom)
            self.assertFalse(ok)
            self.assertIsInstance(exc, OSError)
            self.assertEqual(fr.reason, "released")
            # Reservation no longer holds budget.
            r2 = ledger.try_reserve_ordinary(
                **_reserve_kwargs(target_idx=2, budget=1, min_distinct=1)
            )
            self.assertTrue(r2.ok, r2.reason)


if __name__ == "__main__":
    unittest.main()
