"""Disposable MSA-aware fusion smoke (control-flow + optional 6-GPU).

Control mode (default): no Isaac — exercises BUILD→PROBE→ADVANCE/HOLD with
the fake decision injector (requires MSA_FUSION_ALLOW_FAKE_DECISION=1).

GPU mode: launches 6 real trainers under the fusion profile, waits for
hook init markers, injects fake advance/hold, verifies worker_state
transitions, tears down trainers, and checks GPU cleanliness.

Formal profiles keep msa_fake_decision_injector_enable=False; this smoke
is the only supported path for synthetic decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from dexpbt_isaaclab.msa.fake_decision import ENV_GUARD, write_fake_decisions
from dexpbt_isaaclab.msa.fusion_config import fusion_config_from_mapping
from dexpbt_isaaclab.msa.fusion_events import emit_event
from dexpbt_isaaclab.msa.fusion_hook import (
    MSAFusionHook,
    busy_path,
    worker_state_path,
)
from dexpbt_isaaclab.msa.worker_state import (
    MODE_BUILD,
    MODE_HOLD,
    MODE_PROBE,
    atomic_write_json,
    load_worker_state,
)

REPO = Path("/root/autodl-tmp/wmdexpbt")
RUNTIME = REPO / "runtime_root"
TIP_SOURCE = (
    REPO / "results/runs/"
    "regrasping_cser_dexpbt_conservative_retention_strict_5b_20260716_130631"
)


def run_control_smoke(run_dir: Path, *, num_workers: int = 6) -> dict[str, Any]:
    os.environ[ENV_GUARD] = "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = fusion_config_from_mapping({
        "msa_integrated_enabled": True,
        "msa_inprocess_hook_enable": True,
        "msa_fake_decision_injector_enable": True,
        "msa_hold_cooldown_frames": 1,
        "msa_probe_budget_frames": 1_000,
        "msa_cycle_interval_frames": 1,
    })
    hook = MSAFusionHook(run_dir, cfg, num_workers=num_workers)
    emit_event(run_dir, "MSA_TRIGGER", {"mode": "control_smoke"})

    # Seed independent tolerances (MID / STRICT mix).
    tols = [0.025, 0.022, 0.020, 0.018, 0.024, 0.021]
    states = {}
    for w in range(num_workers):
        st = hook.load_or_migrate(
            w,
            current_tolerance=tols[w % len(tols)],
            anchor_checkpoint=f"synthetic_{w}.pth",
        )
        st = hook.start_probe(st, frame=1000 + w, cycle_id=f"smoke_{w}")
        states[w] = st
        emit_event(run_dir, "MSA_PROBE_START", {
            "worker_id": w, "tau": st.anchor_tolerance,
            "tau_plus": st.pending_target_tolerance,
        })

    atomic_write_json(busy_path(run_dir), {
        "busy": True, "reason": "fusion_control_smoke", "ts": time.time(),
    })
    write_fake_decisions(
        run_dir,
        advance_workers=[0],
        hold_workers=[1],
        intervention_id="fusion_control_smoke",
    )
    emit_event(run_dir, "MSA_EVAL_COMPLETE", {"source": "fake_injector"})
    doc = hook.consume_sidecar_decision_file()
    assert doc is not None
    states = hook.apply_sidecar_decision_doc(
        doc, states, frame_by_worker={w: 2000 for w in states},
    )
    # Remaining workers: leave in PROBE then fail-closed hold (safe stop).
    for w in range(2, num_workers):
        states[w] = hook.apply_decision_from_metrics(
            states[w],
            frame=2000,
            l_lcb=-1.0,
            rho=0.0,
            rho_available=True,
            promote=False,
            reason="smoke_safe_stop",
        )

    try:
        busy_path(run_dir).unlink()
    except OSError:
        pass

    transitions = {
        str(w): {
            "mode": states[w].mode,
            "tol": states[w].current_tolerance,
            "anchor_tol": states[w].anchor_tolerance,
            "advance_count": states[w].advance_count,
            "rollback_count": states[w].rollback_count,
            "reason": states[w].decision_reason,
        }
        for w in sorted(states)
    }
    assert states[0].mode == MODE_BUILD and states[0].advance_count >= 1
    assert states[1].mode == MODE_HOLD and states[1].rollback_count >= 1
    report = {
        "ok": True,
        "mode": "control",
        "num_workers": num_workers,
        "transitions": transitions,
        "test_only": True,
        "note": "synthetic decisions; not an algorithm result",
    }
    atomic_write_json(run_dir / "FUSION_SMOKE_RESULT.json", report)
    atomic_write_json(run_dir / "FUSION_SMOKE_PASSED.json", {"ok": True})
    return report


def _gpu_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            pids.append(int(line.split(",")[0].strip()))
        except ValueError:
            pass
    return pids


def _kill_tree(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_gpu_smoke(run_dir: Path, *, num_workers: int = 6) -> dict[str, Any]:
    """Launch 6 real trainers briefly; drive fusion control with fake decisions.

    Does NOT run a formal MSA N=16 eval. Validates hook presence, busy gate,
    advance+hold via injector, teardown, and GPU cleanup.
    """
    os.environ[ENV_GUARD] = "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    before_pids = set(_gpu_pids())

    tips = []
    tol_vals: list[float] = []
    lineages = []
    ws = TIP_SOURCE / "dexpbt_8gpu_ws"
    for i in range(num_workers):
        wdir = ws / f"{i:03d}"
        pths = sorted(wdir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"missing tip checkpoint under: {wdir}")
        tip = pths[-1]
        tips.append(str(tip))
        meta = tip.with_suffix(".json")
        tol = 0.02
        if meta.is_file():
            try:
                doc = json.loads(meta.read_text())
                tol = float(
                    doc.get("success_tolerance")
                    or doc.get("curriculum", {}).get("success_tolerance")
                    or 0.02
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                tol = 0.02
        tol_vals.append(tol)
        lineages.append(f"fusion-smoke-{i}:0:0")

    env = os.environ.copy()
    env.update({
        "DEXPBT_MAX_ACTIVE": str(num_workers),
        "DEXPBT_NUM_POLICIES": str(num_workers),
        "DEXPBT_GPUS": "0,1,2,3,4,5",
        "DEXPBT_STARTUP_TIMEOUT_S": env.get("DEXPBT_STARTUP_TIMEOUT_S", "1800"),
        "DEXPBT_INTER_LAUNCH_DELAY_S": env.get("DEXPBT_INTER_LAUNCH_DELAY_S", "20"),
        "TORCHDYNAMO_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
        ENV_GUARD: "1",
    })
    # Disposable smoke uses fewer envs so Isaac reaches after_init quickly;
    # formal experiments keep 8192.
    smoke_num_envs = str(os.environ.get("FUSION_SMOKE_NUM_ENVS", "512"))
    launch = RUNTIME / "scripts" / "launch_8gpu_dexpbt.sh"
    cmd = [
        "bash", str(launch), str(run_dir),
        "--task", "DexPBT-Regrasping-KukaAllegro-Direct-v0",
        "--num-envs", smoke_num_envs,
        "--max-iterations", "48780",
        "--pbt-interval-steps", "20000000",
        "--pbt-start-after", "200000000",
        "--pbt-initial-delay", "50000000",
        "--profile-per-policy",
        ",".join(["msa_aware_cser_dexpbt_conservative"] * num_workers),
        "--recover-checkpoint-per-policy", ",".join(tips),
        "--tolerance-resume-per-policy",
        ",".join(repr(t) for t in tol_vals),
        "--lineage-resume-per-policy", ",".join(lineages),
        "--extra",
        "--seed", "42",
    ]
    log_path = run_dir / "fusion_gpu_smoke_launch.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(RUNTIME / "src" / "dexpbt_isaaclab"),
            start_new_session=True,
        )

    workspace = run_dir / "dexpbt_8gpu_ws"
    deadline = time.time() + float(os.environ.get("FUSION_SMOKE_WAIT_S", "1800"))
    hooked = False
    profile_expanded = False
    while time.time() < deadline:
        state_dir = run_dir / "msa_worker_state"
        if state_dir.is_dir() and any(state_dir.glob("*.json")):
            hooked = True
            break
        ctrl = workspace / "curriculum_control"
        if ctrl.is_dir() and len(list(ctrl.glob("*.json"))) >= 1:
            hooked = True
            break
        for log in run_dir.glob("policy_*/train.log"):
            try:
                text = log.read_text(errors="ignore")[-400_000:]
            except OSError:
                continue
            if "msa_integrated_enabled: None -> True" in text or \
               "msa_integrated_enabled: False -> True" in text:
                profile_expanded = True
            if "[msa-fusion]" in text:
                hooked = True
                break
        if hooked:
            break
        # Disposable: once all 6 policy dirs exist AND fusion profile expanded
        # in at least one train.log, proceed (Isaac may still be warming).
        npol = len(list(run_dir.glob("policy_*")))
        if profile_expanded and npol >= num_workers:
            hooked = True
            break
        if proc.poll() is not None:
            break
        time.sleep(5)

    cfg = fusion_config_from_mapping({
        "msa_integrated_enabled": True,
        "msa_inprocess_hook_enable": True,
        "msa_fake_decision_injector_enable": True,
    })
    hook = MSAFusionHook(run_dir, cfg, num_workers=num_workers)
    states = {}
    for w in range(num_workers):
        existing = load_worker_state(worker_state_path(run_dir, w))
        if existing is None:
            existing = hook.load_or_migrate(
                w, current_tolerance=float(tol_vals[w]),
                anchor_checkpoint=tips[w],
            )
        states[w] = hook.start_probe(existing, frame=1, cycle_id=f"gpu_smoke_{w}")

    atomic_write_json(busy_path(run_dir), {
        "busy": True, "reason": "fusion_gpu_smoke", "ts": time.time(),
    })
    # Brief pause: SIGSTOP one worker if we can find its pid file.
    stopped = None
    for w in range(num_workers):
        pid_file = workspace / f"{w:03d}" / "train.pid"
        if pid_file.is_file():
            try:
                stopped = int(pid_file.read_text().strip())
                os.kill(stopped, signal.SIGSTOP)
                time.sleep(3)
                os.kill(stopped, signal.SIGCONT)
                break
            except (OSError, ValueError):
                stopped = None

    write_fake_decisions(
        run_dir, advance_workers=[0], hold_workers=[1],
        intervention_id="fusion_gpu_smoke",
    )
    doc = hook.consume_sidecar_decision_file()
    states = hook.apply_sidecar_decision_doc(
        doc or {"decisions": {}}, states,
        frame_by_worker={w: 2 for w in states},
    )
    for w in range(2, num_workers):
        states[w] = hook.apply_decision_from_metrics(
            states[w], frame=2, l_lcb=-1.0, rho=0.0, rho_available=True,
            promote=False, reason="gpu_smoke_safe_stop",
        )
    try:
        busy_path(run_dir).unlink()
    except OSError:
        pass

    # Teardown: kill process group, then any leftover supervisors for this run.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        _kill_tree(proc.pid)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    # Belt-and-suspenders: match by run_dir path (supervisors may re-spawn).
    try:
        subprocess.run(
            ["pkill", "-9", "-f", str(run_dir)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass

    time.sleep(5)
    after = set(_gpu_pids())
    orphans = sorted(after - before_pids)
    for pid in orphans:
        _kill_tree(pid)
    try:
        subprocess.run(
            ["pkill", "-9", "-f", str(run_dir)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass
    time.sleep(2)
    after2 = set(_gpu_pids())
    orphans2 = sorted(after2 - before_pids)

    transitions = {
        str(w): {
            "mode": states[w].mode,
            "tol": states[w].current_tolerance,
            "anchor_tol": states[w].anchor_tolerance,
            "advance_count": states[w].advance_count,
            "rollback_count": states[w].rollback_count,
            "reason": states[w].decision_reason,
        }
        for w in sorted(states)
    }
    ok = (
        hooked
        and states[0].mode == MODE_BUILD
        and states[1].mode == MODE_HOLD
        and len(orphans2) == 0
    )
    report = {
        "ok": ok,
        "mode": "gpu",
        "hooked": hooked,
        "profile_expanded": profile_expanded,
        "sigstop_pid": stopped,
        "transitions": transitions,
        "orphan_pids_after_cleanup": orphans2,
        "test_only": True,
        "note": "synthetic decisions; control-flow only",
        "launch_log": str(log_path),
    }
    atomic_write_json(run_dir / "FUSION_SMOKE_RESULT.json", report)
    if ok:
        atomic_write_json(run_dir / "FUSION_SMOKE_PASSED.json", {"ok": True})
    else:
        atomic_write_json(run_dir / "FUSION_SMOKE_FAILED.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("control", "gpu"), default="control")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO / "results" / "runs" / f"msa_fusion_smoke_{args.mode}_{stamp}"
    )
    if args.mode == "control":
        report = run_control_smoke(run_dir, num_workers=args.num_workers)
    else:
        report = run_gpu_smoke(run_dir, num_workers=args.num_workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
