"""Tests for the state-faithful bootstrap of the stateful MSA arms.

Covers:
- state_bootstrap: copy + rebuild + fail-closed + STATE_FIDELITY +
  fingerprint verification + committed-cadence audit;
- observer seed loading (pbt_observer._maybe_apply_observer_state_seed);
- continuation_guard: structured RESUME_APPLIED validation, control
  fidelity checks, the stall reproduction gate, escalating stops;
- launch_pbt env-clock resume parsing and arms clock-list generation.

CPU-only; no Isaac, no GPU.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path

from dexpbt_isaaclab.msa import state_bootstrap as sb
from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import (
    CurriculumControlState,
    read_curriculum_control,
    write_curriculum_control,
)

TIP_FRAME = 5_000_000_000
TICK = 80_000_000
SRC_TOL = 0.023112487509668322


def _make_source_run(root: Path, num_workers: int = 6,
                     n_metas: int = 14) -> Path:
    """Synthetic source run with realistic control JSONs, resume snapshots,
    committed metas at the real 80M cadence and LATEST.commit files."""
    src = root / "source_run"
    for w in range(num_workers):
        ws = src / "dexpbt_8gpu_ws" / f"{w:03d}"
        ws.mkdir(parents=True)
        # committed metas: 80M cadence ending exactly at the 5B tip; one
        # rollback (tolerance increase) then two shrinks inside the window
        frames = [TIP_FRAME - (n_metas - 1 - i) * TICK
                  for i in range(n_metas)]
        tols = [SRC_TOL * 1.05] * (n_metas - 3) + [SRC_TOL * 1.02,
                                                   SRC_TOL * 1.01, SRC_TOL]
        for i, (frame, tol) in enumerate(zip(frames, tols)):
            prefix = f"{160 - (n_metas - 1 - i):06d}"
            (ws / f"{prefix}.json").write_text(json.dumps({
                "frame": frame, "iteration": 160 - (n_metas - 1 - i),
                "lineage_id": f"lin{w}",
                "nsucc_mean": 30.0 + i * 0.1, "nsucc_max": 45.0,
                "curriculum": {"success_tolerance": tol},
            }))
            (ws / f"{prefix}.pth").write_bytes(b"ckpt")
        (ws / "LATEST.commit").write_text(json.dumps(
            {"commit_dir": "000160.pth", "commit_id": "c-160"}))
        # control JSON with non-default gate state (the stall)
        control = CurriculumControlState(
            allow_shrink=False, shrink_token=False,
            shrink_blocked_reason="post_shrink_consolidation",
            post_shrink_consolidation_until_restart_step=716_224,
            collapse_freeze_active=False,
            collapse_freeze_until_env_frames=4_900_000_000,
            recent_peak_nsucc_mean=33.23,
            last_mastered_tolerance=SRC_TOL,
            curriculum_last_mastered_tolerance=SRC_TOL * 1.05,
            curriculum_last_mastered_frame=4_800_000_000,
            curriculum_last_mastered_nsucc_mean=32.0,
            curriculum_last_mastered_global_retention=0.97,
            curriculum_repeated_collapse_count=1,
            rollback_count=2, last_rollback_frame=frames[-4],
            rollback_reason="repeated_collapse_to_last_mastered",
            curriculum_rollback_freeze_until=5_050_000_000,
            late_floor_retention_freeze_until=5_100_000_000,
            wm_curriculum_freeze_until=4_950_000_000,
            floor_recovery_protect_until=5_020_000_000,
            first_floor_frame=3_000_000_000,
            first_hard_floor_frame=3_200_000_000,
            # stale msa state that MUST be cleared on restore
            msa_enabled=True, msa_forced_success_tolerance=0.0219,
            msa_cycle={"status": "DONE"},
        )
        ctl_dir = src / "dexpbt_8gpu_ws" / "curriculum_control"
        ctl_dir.mkdir(parents=True, exist_ok=True)
        write_curriculum_control(ctl_dir / f"{w:03d}.json", control)
        # resume snapshot (env clock)
        pol = src / f"policy_{w:02d}"
        pol.mkdir(parents=True)
        (pol / "curriculum_resume_snapshot.json").write_text(json.dumps({
            "schema_version": 1, "policy_idx": w, "env_frames": TIP_FRAME,
            "curriculum": {
                "success_tolerance": SRC_TOL,
                "initial_tolerance": 0.075, "target_tolerance": 0.0075,
                "last_curriculum_update": 691_809,
                "frame_since_restart": 711_360,
                "curriculum_freeze_frames_after_restart": 0,
            }}))
    return src


class TestStateBootstrap(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)
        self.src = _make_source_run(self.root)
        self.run_dir = self.root / "new_run"
        self.ws = self.run_dir / "dexpbt_8gpu_ws"
        self.run_dir.mkdir()

    def _bootstrap(self):
        return sb.bootstrap_run(self.src, self.run_dir, self.ws)

    def test_control_copied_gate_state_intact_msa_cleared(self):
        self._bootstrap()
        src_ctl = read_curriculum_control(
            self.src / "dexpbt_8gpu_ws" / "curriculum_control" / "000.json")
        new_ctl = read_curriculum_control(
            self.ws / "curriculum_control" / "000.json")
        # every base-retention gate field copied EXACTLY
        s, n = asdict(src_ctl), asdict(new_ctl)
        for k in s:
            if k.startswith("msa_"):
                continue
            self.assertEqual(s[k], n[k], f"gate field {k} not copied")
        # stale msa_* cleared to defaults
        self.assertFalse(n["msa_enabled"])
        self.assertEqual(n["msa_forced_success_tolerance"], 0.0)
        self.assertIsNone(n["msa_cycle"])

    def test_observer_seed_rebuilt_from_metas_and_control(self):
        self._bootstrap()
        seed = json.loads(
            (self.ws / "observer_state_seed" / "000.json").read_text())
        self.assertEqual(len(seed["window_samples"]), sb.WINDOW_MAX_LEN)
        last = seed["window_samples"][-1]
        self.assertEqual(last["env_frames"], TIP_FRAME)
        self.assertAlmostEqual(last["success_tolerance"], SRC_TOL, places=12)
        self.assertAlmostEqual(seed["recent_peak_nsucc_mean"], 33.23)
        self.assertAlmostEqual(seed["last_mastered_tolerance"], SRC_TOL,
                               places=12)
        # safety freeze = max(rollback, late-floor) horizons
        self.assertEqual(seed["curriculum_safety_freeze_until"],
                         5_100_000_000)
        self.assertEqual(seed["seed_valid_until_frame"],
                         TIP_FRAME + 2 * TICK)

    def test_shrinks_since_rollback_counts_only_post_rollback(self):
        self._bootstrap()
        seed = json.loads(
            (self.ws / "observer_state_seed" / "000.json").read_text())
        # source history: rollback frame = frames[-4]; three shrink
        # transitions land after it (1.05->1.02->1.01->1.0 x SRC_TOL)
        self.assertEqual(seed["shrinks_since_rollback"], 3)
        # shrinks at/before the rollback frame are excluded
        self.assertEqual(sb.rebuild_shrinks_since_rollback(
            [{"frame": 100, "curriculum": {"success_tolerance": 0.030}},
             {"frame": 200, "curriculum": {"success_tolerance": 0.028}},
             {"frame": 300, "curriculum": {"success_tolerance": 0.026}}],
            last_rollback_frame=200), 1)

    def test_fidelity_report_faithful_with_explicit_losses(self):
        doc = self._bootstrap()
        self.assertTrue(doc["state_faithful"])
        w0 = doc["workers"][0]
        lost = [k for k, v in w0["fields"].items()
                if v["category"] == "default_lost"]
        self.assertTrue(all(not w0["fields"][k]["match"] for k in lost))
        # non-lost critical fields all match
        for k, v in w0["fields"].items():
            if v["category"] != "default_lost":
                self.assertTrue(v["match"], f"{k} mismatched")
        # env clock exported for the CLI resume flags
        self.assertEqual(doc["env_clocks"]["0"]["last_curriculum_update"],
                         691_809)
        self.assertEqual(doc["env_clocks"]["0"]["frame_since_restart"],
                         711_360)

    def test_resume_snapshot_copied_into_new_policy_dirs(self):
        self._bootstrap()
        for w in range(6):
            p = (self.run_dir / f"policy_{w:02d}"
                 / "curriculum_resume_snapshot.json")
            self.assertTrue(p.is_file())
            snap = json.loads(p.read_text())
            self.assertEqual(snap["curriculum"]["success_tolerance"],
                             SRC_TOL)

    def test_fail_closed_on_missing_control(self):
        (self.src / "dexpbt_8gpu_ws" / "curriculum_control"
         / "003.json").unlink()
        with self.assertRaises(sb.BootstrapError):
            self._bootstrap()
        # nothing claims faithfulness
        self.assertFalse((self.run_dir / sb.FIDELITY_NAME).exists())

    def test_fail_closed_on_short_meta_history(self):
        ws3 = self.src / "dexpbt_8gpu_ws" / "003"
        for p in sorted(ws3.glob("0*.json"))[:-5]:
            p.unlink()
        with self.assertRaises(sb.BootstrapError) as cm:
            self._bootstrap()
        self.assertIn("committed metas", str(cm.exception))

    def test_fingerprint_verify_and_tamper_detection(self):
        self._bootstrap()
        self.assertEqual(sb.verify_fingerprint(self.run_dir, self.ws), [])
        # tamper the RESTORED state -> fingerprint mismatch
        ctl_path = self.ws / "curriculum_control" / "002.json"
        ctl = read_curriculum_control(ctl_path)
        ctl.last_mastered_tolerance = 0.075
        write_curriculum_control(ctl_path, ctl)
        fails = sb.verify_fingerprint(self.run_dir, self.ws)
        self.assertTrue(any("fingerprint mismatch" in f for f in fails))

    def test_fingerprint_detects_source_change(self):
        self._bootstrap()
        snap = self.src / "policy_00" / "curriculum_resume_snapshot.json"
        data = json.loads(snap.read_text())
        data["curriculum"]["success_tolerance"] = 0.05
        snap.write_text(json.dumps(data))
        fails = sb.verify_fingerprint(self.run_dir, self.ws)
        self.assertTrue(any("source file CHANGED" in f for f in fails))

    def test_cadence_audit_measures_real_80m(self):
        audit = sb.cadence_audit(self.src)
        self.assertEqual(audit["committed_tick_frames"], TICK)
        self.assertTrue(audit["matches_module_constant"])
        for w in range(6):
            self.assertEqual(audit["workers"][str(w)]["tail_cadence_frames"],
                             TICK)


class TestObserverSeedLoading(unittest.TestCase):
    """The observer restores its in-memory curriculum state from
    observer_state_seed/ exactly once, only inside the validity window and
    never regressing live state."""

    def _bare_observer(self, workspace: Path, num_policies: int = 6):
        from dexpbt_isaaclab.pbt.pbt_observer import DexPBTPbtAlgoObserver
        obs = DexPBTPbtAlgoObserver.__new__(DexPBTPbtAlgoObserver)
        obs.workspace = workspace
        obs.num_policies = num_policies
        obs._observer_seed_checked = False
        obs._policy_nsucc_windows = {}
        obs._recent_peak_nsucc_mean = {}
        obs._collapse_freeze_until = {}
        obs._curriculum_safety_freeze_until = {}
        obs._wm_curriculum_freeze_until = {}
        obs._last_mastered_tolerance = {}
        obs._last_mastered_safe_tolerance = {}
        obs._last_mastered_safe_frame = {}
        obs._last_mastered_safe_nsucc_mean = {}
        obs._last_mastered_safe_global_retention = {}
        obs._repeated_collapse_count = {}
        obs._rollback_count = {}
        obs._last_rollback_frame = {}
        obs._rollback_reason = {}
        obs._shrinks_since_rollback = {}
        obs._floor_recovery_protect_until = {}
        obs._first_floor_frame = {}
        obs._first_hard_floor_frame = {}
        return obs

    def _seeded_workspace(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        src = _make_source_run(root)
        run_dir = root / "run"
        ws = run_dir / "dexpbt_8gpu_ws"
        run_dir.mkdir()
        sb.bootstrap_run(src, run_dir, ws)
        return ws

    def test_seed_restores_windows_and_gate_memory(self):
        ws = self._seeded_workspace()
        obs = self._bare_observer(ws)
        obs._maybe_apply_observer_state_seed(TIP_FRAME + TICK)
        for w in range(6):
            window = obs._policy_nsucc_windows[w]
            self.assertEqual(len(window.samples), sb.WINDOW_MAX_LEN)
            self.assertEqual(window.samples[-1].env_frames, TIP_FRAME)
            self.assertAlmostEqual(window.samples[-1].success_tolerance,
                                   SRC_TOL, places=12)
            self.assertAlmostEqual(obs._recent_peak_nsucc_mean[w], 33.23)
            self.assertAlmostEqual(obs._last_mastered_tolerance[w], SRC_TOL,
                                   places=12)
            self.assertEqual(obs._collapse_freeze_until[w], 4_900_000_000)
            self.assertEqual(obs._curriculum_safety_freeze_until[w],
                             5_100_000_000)
            self.assertEqual(obs._rollback_count[w], 2)
            self.assertEqual(obs._shrinks_since_rollback[w], 3)
            self.assertEqual(obs._repeated_collapse_count[w], 1)

    def test_stale_seed_skipped_after_validity_window(self):
        ws = self._seeded_workspace()
        obs = self._bare_observer(ws)
        obs._maybe_apply_observer_state_seed(TIP_FRAME + 3 * TICK)
        self.assertEqual(obs._policy_nsucc_windows, {})
        self.assertEqual(obs._recent_peak_nsucc_mean, {})

    def test_live_state_never_regressed(self):
        from dexpbt_isaaclab.pbt.pbt_curriculum_p1 import WindowSample
        ws = self._seeded_workspace()
        obs = self._bare_observer(ws)
        live = obs._policy_window(0)
        live.append(WindowSample(env_frames=TIP_FRAME + TICK,
                                 nsucc_mean=10.0, nsucc_max=20.0,
                                 success_tolerance=0.02, pbt_iteration=161,
                                 lineage_id="lin0"))
        obs._maybe_apply_observer_state_seed(TIP_FRAME + TICK)
        # policy 0 untouched (already live); policy 1 seeded
        self.assertEqual(len(obs._policy_nsucc_windows[0].samples), 1)
        self.assertEqual(len(obs._policy_nsucc_windows[1].samples),
                         sb.WINDOW_MAX_LEN)

    def test_noop_without_seed_dir(self):
        with tempfile.TemporaryDirectory() as td:
            obs = self._bare_observer(Path(td))
            obs._maybe_apply_observer_state_seed(TIP_FRAME)
            self.assertEqual(obs._policy_nsucc_windows, {})
            self.assertTrue(obs._observer_seed_checked)

    def test_applied_once_per_process(self):
        ws = self._seeded_workspace()
        obs = self._bare_observer(ws)
        obs._maybe_apply_observer_state_seed(TIP_FRAME + TICK)
        obs._policy_nsucc_windows.clear()
        obs._maybe_apply_observer_state_seed(TIP_FRAME + TICK)
        self.assertEqual(obs._policy_nsucc_windows, {})  # checked once


def _faithful_seed() -> dict:
    return {
        "last_mastered_tolerance": SRC_TOL,
        "recent_peak_nsucc_mean": 33.23,
        "last_mastered_safe_tolerance": SRC_TOL * 1.05,
        "collapse_freeze_until": 4_900_000_000,
    }


class TestGuardControlFidelity(unittest.TestCase):
    def test_faithful_control_passes(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            verify_control_fidelity,
        )
        control = CurriculumControlState(
            last_mastered_tolerance=SRC_TOL,
            recent_peak_nsucc_mean=33.5,        # new peak is fine
            curriculum_last_mastered_tolerance=SRC_TOL * 1.05,
            collapse_freeze_until_env_frames=4_900_000_000)
        self.assertEqual(
            verify_control_fidelity(control, _faithful_seed()), [])

    def test_default_clobber_detected(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            verify_control_fidelity,
        )
        # the exact production hazard: last_mastered clobbered to 0.075,
        # peak map reset, freeze horizon zeroed
        control = CurriculumControlState(
            last_mastered_tolerance=0.075,
            recent_peak_nsucc_mean=0.0,
            curriculum_last_mastered_tolerance=0.0,
            collapse_freeze_until_env_frames=0)
        violations = verify_control_fidelity(control, _faithful_seed())
        self.assertTrue(any("last_mastered_tolerance" in v
                            for v in violations))
        self.assertTrue(any("recent_peak" in v for v in violations))
        self.assertTrue(any("freeze horizon" in v for v in violations))

    def test_missing_control_fails(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            verify_control_fidelity,
        )
        self.assertTrue(verify_control_fidelity(None, _faithful_seed()))


class TestStallReproductionGate(unittest.TestCase):
    def _records(self, reproduced_count: int) -> dict:
        recs = {}
        for w in range(6):
            recs[w] = {"stall_reproduced": w < reproduced_count,
                       "validated_tolerance": SRC_TOL
                       if w < reproduced_count else 0.95 * SRC_TOL}
        return recs

    def test_four_of_six_passes(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            stall_reproduction_gate,
        )
        self.assertTrue(stall_reproduction_gate(self._records(4)).ok)
        self.assertTrue(stall_reproduction_gate(self._records(6)).ok)

    def test_three_of_six_fails(self):
        from dexpbt_isaaclab.msa.continuation_guard import (
            stall_reproduction_gate,
        )
        v = stall_reproduction_gate(self._records(3))
        self.assertFalse(v.ok)
        self.assertIn("stall reproduction FAILED", v.reason)


class TestGuardLoopStateFaithful(unittest.TestCase):
    """End-to-end guard loop on a state-faithful arm: structured
    RESUME_APPLIED, control fidelity, stall reproduction, markers, and
    the true readiness barrier (workers frozen while validating)."""

    def _reap(self, proc) -> None:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    def _setup_run(self, root: Path, tols: dict[int, float],
                   with_resume: bool = True):
        run_dir = root / "run"
        ws = run_dir / "dexpbt_8gpu_ws"
        procs = {}
        for w in range(6):
            d = ws / f"{w:03d}"
            d.mkdir(parents=True)
            (d / "000161.pth").write_bytes(b"ckpt")
            (d / "000161.json").write_text(json.dumps({
                "frame": TIP_FRAME + TICK, "lineage_id": f"lin{w}",
                "curriculum": {"success_tolerance": tols[w]}}))
            seed_dir = ws / "observer_state_seed"
            seed_dir.mkdir(exist_ok=True)
            (seed_dir / f"{w:03d}.json").write_text(
                json.dumps(_faithful_seed()))
            ctl_dir = ws / "curriculum_control"
            ctl_dir.mkdir(exist_ok=True)
            write_curriculum_control(
                ctl_dir / f"{w:03d}.json",
                CurriculumControlState(
                    last_mastered_tolerance=SRC_TOL,
                    recent_peak_nsucc_mean=33.23,
                    curriculum_last_mastered_tolerance=SRC_TOL * 1.05,
                    collapse_freeze_until_env_frames=4_900_000_000))
            pol = run_dir / f"policy_{w:02d}"
            pol.mkdir(parents=True)
            # a REAL process per worker: the readiness barrier SIGSTOPs it
            proc = subprocess.Popen(["sleep", "300"])
            self.addCleanup(self._reap, proc)
            (pol / "pid.txt").write_text(str(proc.pid))
            procs[w] = proc
            if with_resume:
                (pol / "RESUME_APPLIED.json").write_text(json.dumps({
                    "policy_idx": w,
                    "applied": {"success_tolerance": SRC_TOL}}))
        config = {"state_faithful": True, "resume_checkpoints": {
            f"p{w}": {"frame": TIP_FRAME, "live_tolerance": SRC_TOL,
                      "lineage_id": f"lin{w}", "iteration": 160,
                      "path": f"/x/{w:03d}/000160.pth", "sha256_head": "h"}
            for w in range(6)}}
        return run_dir, ws, config

    def test_full_pass_writes_marker_with_reproduction(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            run_dir, ws, config = self._setup_run(
                Path(td), {w: SRC_TOL for w in range(6)})
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 0)
            marker = json.loads(
                (run_dir / "FIRST_TICK_VALIDATED.json").read_text())
            self.assertTrue(marker["stall_reproduction"]["passed"])
            self.assertEqual(
                marker["stall_reproduction"]["reproduced_workers"],
                list(range(6)))
            for rec in marker["workers"]:
                self.assertTrue(rec["stall_reproduced"])
            self.assertFalse((run_dir / "GUARD_FAILED").exists())
            self.assertFalse(
                (run_dir / "STATE_FIDELITY_FAILED.json").exists())

    def test_one_legal_shrink_still_passes_gate(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            tols = {w: SRC_TOL for w in range(6)}
            tols[5] = 0.951 * SRC_TOL        # one legal shrink, 5/6 held
            run_dir, ws, config = self._setup_run(Path(td), tols)
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 0)
            marker = json.loads(
                (run_dir / "FIRST_TICK_VALIDATED.json").read_text())
            self.assertTrue(marker["stall_reproduction"]["passed"])
            self.assertNotIn(
                5, marker["stall_reproduction"]["reproduced_workers"])

    def test_majority_breakout_writes_state_fidelity_failed(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            tols = {w: 0.951 * SRC_TOL for w in range(6)}  # all moved
            tols[0] = SRC_TOL
            tols[1] = SRC_TOL                              # only 2/6 held
            run_dir, ws, config = self._setup_run(Path(td), tols)
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 2)
            self.assertTrue(
                (run_dir / "STATE_FIDELITY_FAILED.json").exists())
            self.assertFalse(
                (run_dir / "FIRST_TICK_VALIDATED.json").exists())
            payload = json.loads(
                (run_dir / "STATE_FIDELITY_FAILED.json").read_text())
            self.assertIn("stall reproduction FAILED", payload["reason"])

    def test_missing_resume_applied_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            run_dir, ws, config = self._setup_run(
                Path(td), {w: SRC_TOL for w in range(6)}, with_resume=False)
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 2)
            self.assertTrue((run_dir / "GUARD_FAILED").exists())
            reason = json.loads(
                (run_dir / "GUARD_FAILED").read_text())["reason"]
            self.assertIn("RESUME_APPLIED", reason)

    def test_legacy_cascade_fails_closed(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            tols = {w: SRC_TOL for w in range(6)}
            tols[0] = 0.016849       # the real 2026-07-19 failure value
            run_dir, ws, config = self._setup_run(Path(td), tols)
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 2)
            reason = json.loads(
                (run_dir / "GUARD_FAILED").read_text())["reason"]
            self.assertIn("legacy", reason)

    def test_clobbered_control_writes_state_fidelity_failed(self):
        from dexpbt_isaaclab.msa.continuation_guard import guard_loop
        with tempfile.TemporaryDirectory() as td:
            run_dir, ws, config = self._setup_run(
                Path(td), {w: SRC_TOL for w in range(6)})
            write_curriculum_control(
                ws / "curriculum_control" / "002.json",
                CurriculumControlState())   # defaults: the clobber hazard
            rc = guard_loop(run_dir, ws, config, max_iterations=1)
            self.assertEqual(rc, 2)
            payload = json.loads(
                (run_dir / "STATE_FIDELITY_FAILED.json").read_text())
            self.assertEqual(payload["worker_id"], 2)
            self.assertTrue(payload["violations"])


class TestEscalatingStop(unittest.TestCase):
    """stop chain: SIGCONT -> SIGTERM -> grace -> re-enumerate -> SIGKILL
    -> verify exit (the smoke showed Isaac trainers ignoring SIGTERM)."""

    def test_sigterm_ignoring_process_is_sigkilled(self):
        from dexpbt_isaaclab.msa.continuation_guard import _terminate_tree
        proc = subprocess.Popen(
            ["bash", "-c", "trap '' TERM; sleep 300"])
        self.addCleanup(lambda: proc.poll() or proc.kill())
        time.sleep(0.3)                        # let the trap install
        left = _terminate_tree(proc.pid, grace_s=1.0, verify_s=5.0)
        self.assertEqual(left, [])
        deadline = time.monotonic() + 5
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(proc.poll(), -signal.SIGKILL)

    def test_cooperative_process_exits_on_sigterm(self):
        from dexpbt_isaaclab.msa.continuation_guard import _terminate_tree
        proc = subprocess.Popen(["sleep", "300"])
        self.addCleanup(lambda: proc.poll() or proc.kill())
        left = _terminate_tree(proc.pid, grace_s=5.0, verify_s=5.0)
        self.assertEqual(left, [])
        deadline = time.monotonic() + 5
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(proc.poll(), -signal.SIGTERM)

    def test_stopped_process_receives_sigcont_first(self):
        from dexpbt_isaaclab.msa.continuation_guard import _terminate_tree
        proc = subprocess.Popen(["sleep", "300"])
        self.addCleanup(lambda: proc.poll() or proc.kill())
        os.kill(proc.pid, signal.SIGSTOP)
        time.sleep(0.2)
        left = _terminate_tree(proc.pid, grace_s=5.0, verify_s=5.0)
        self.assertEqual(left, [])
        deadline = time.monotonic() + 5
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(proc.poll())


class TestClockResume(unittest.TestCase):
    def test_parse_valid(self):
        from dexpbt_isaaclab.scripts.launch_pbt import (
            parse_clock_resume_per_policy,
        )
        pairs = parse_clock_resume_per_policy(
            "691809:711360,1:2,3:4,5:6,7:8,9:10", 6)
        self.assertEqual(pairs[0], (691_809, 711_360))
        self.assertEqual(len(pairs), 6)

    def test_parse_rejects_bad_shapes(self):
        from dexpbt_isaaclab.scripts.launch_pbt import (
            parse_clock_resume_per_policy,
        )
        with self.assertRaises(ValueError):
            parse_clock_resume_per_policy("1:2,3:4", 6)      # wrong count
        with self.assertRaises(ValueError):
            parse_clock_resume_per_policy(
                "1:2:3,1:2,1:2,1:2,1:2,1:2", 6)              # bad entry
        with self.assertRaises(ValueError):
            parse_clock_resume_per_policy(
                "5:2,1:2,1:2,1:2,1:2,1:2", 6)                # lcu > fsr

    def test_arms_clock_list_from_source(self):
        from dexpbt_isaaclab.msa.arms import _clock_list_from_source
        with tempfile.TemporaryDirectory() as td:
            src = _make_source_run(Path(td))
            clock_list = _clock_list_from_source(src)
            self.assertEqual(clock_list,
                             ",".join(["691809:711360"] * 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
