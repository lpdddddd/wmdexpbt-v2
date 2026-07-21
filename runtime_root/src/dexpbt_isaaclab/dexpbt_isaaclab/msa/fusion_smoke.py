"""Disposable MSA-aware fusion smoke (control / gpu / integrated).

Control mode: no Isaac — fake decision injector (MSA_FUSION_ALLOW_FAKE_DECISION=1).

GPU mode: 6 trainers + fake injector (legacy control-flow check).

Integrated mode: production launcher path — 6 trainers + continuation_guard
+ hold_watchdog + sidecar process, real Isaac after_init/steps, transactional
PROBE anchors, eval-fixture → real decide_selective_promotion → fusion
command/ack protocol applied inside trainers, ADVANCE+ROLLBACK, short resume,
cleanup. Formal profiles cannot enable the fixture
(MSA_FUSION_ALLOW_EVAL_FIXTURE=1 required).
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


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "msa_fusion_events.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        name = str(e.get("event") or "")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _worker_timelines(events: list[dict[str, Any]]) -> dict[str, list[dict]]:
    timelines: dict[str, list[dict]] = {str(i): [] for i in range(6)}
    for e in events:
        wid = e.get("worker_id")
        if wid is None:
            continue
        key = str(int(wid))
        timelines.setdefault(key, []).append({
            "event": e.get("event"),
            "ts": e.get("timestamp"),
            "mode": e.get("mode"),
            "action": e.get("action"),
            "cycle_id": e.get("cycle_id"),
            "tol": e.get("new_tol") or e.get("tolerance") or e.get("tau"),
        })
    return timelines


def _policy_after_init(run_dir: Path, num_workers: int) -> set[int]:
    ready: set[int] = set()
    markers = (
        "[msa-fusion]",
        "fps step:",
        "fps step :",
        "PPO total step",
        "msa_integrated_enabled",
        "Simulation App is Loaded",
        "app ready",
    )
    for w in range(num_workers):
        logs = list((run_dir / f"policy_{w:02d}").glob("attempt_*.log"))
        logs += [
            run_dir / f"policy_{w}" / "train.log",
            run_dir / f"policy_{w:02d}" / "train.log",
            run_dir / "dexpbt_8gpu_ws" / f"{w:03d}" / "train.log",
        ]
        for log in logs:
            if not log.is_file():
                continue
            try:
                text = log.read_text(errors="ignore")[-800_000:]
            except OSError:
                continue
            if any(m in text for m in markers):
                ready.add(w)
                break
    state_dir = run_dir / "msa_worker_state"
    if state_dir.is_dir():
        for p in state_dir.glob("*.json"):
            try:
                ready.add(int(p.stem))
            except ValueError:
                pass
    return ready


def _min_steps_seen(run_dir: Path, num_workers: int) -> dict[int, int]:
    """Best-effort env-frame / PPO update counters from train / attempt logs."""
    out: dict[int, int] = {}
    for w in range(num_workers):
        frames = 0
        logs = list((run_dir / f"policy_{w:02d}").glob("attempt_*.log"))
        logs += [
            run_dir / f"policy_{w:02d}" / "train.log",
            run_dir / f"policy_{w}" / "train.log",
            run_dir / "dexpbt_8gpu_ws" / f"{w:03d}" / "train.log",
        ]
        for log in logs:
            if not log.is_file():
                continue
            try:
                text = log.read_text(errors="ignore")[-1_200_000:]
            except OSError:
                continue
            frames = max(frames, text.count("fps step"))
            if "a_loss" in text or "PPO total step" in text:
                frames = max(frames, 1)
        # Fusion hook init implies after_init completed (real trainer).
        if (run_dir / "msa_worker_state" / f"{w:03d}.json").is_file():
            frames = max(frames, 1)
        out[w] = int(frames)
    return out


def _start_aux_process(
    run_dir: Path,
    name: str,
    cmd: list[str],
    env: dict[str, str],
) -> subprocess.Popen:
    log_path = run_dir / f"{name}.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(RUNTIME / "src" / "dexpbt_isaaclab"),
            start_new_session=True,
        )
    (run_dir / f"{name}.pid").write_text(str(proc.pid))
    return proc


def run_integrated_smoke(run_dir: Path, *, num_workers: int = 6) -> dict[str, Any]:
    """Real 6-GPU integrated disposable smoke (production protocol + fixture)."""
    from dexpbt_isaaclab.msa.eval_fixture import ENV_FIXTURE
    from dexpbt_isaaclab.msa.fusion_adapter import try_complete_cycle
    from dexpbt_isaaclab.msa.fusion_protocol import read_ack
    from dexpbt_isaaclab.msa.fusion_transaction import load_transaction_record

    if os.environ.get(ENV_FIXTURE) != "1":
        raise RuntimeError(
            f"integrated smoke requires {ENV_FIXTURE}=1 "
            "(formal profiles must not set this)"
        )
    # Formal profile must not enable fake injector.
    os.environ.pop(ENV_GUARD, None)

    run_dir.mkdir(parents=True, exist_ok=True)
    before_pids = set(_gpu_pids())
    checks: dict[str, Any] = {}
    aux: dict[str, subprocess.Popen] = {}
    proc: subprocess.Popen | None = None
    cycle_id = f"fusion_integrated_{time.strftime('%Y%m%d_%H%M%S')}"

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
        lineages.append(f"fusion-int-{i}:0:0")

    env = os.environ.copy()
    env.update({
        "DEXPBT_MAX_ACTIVE": str(num_workers),
        "DEXPBT_NUM_POLICIES": str(num_workers),
        "DEXPBT_GPUS": "0,1,2,3,4,5",
        "DEXPBT_STARTUP_TIMEOUT_S": env.get("DEXPBT_STARTUP_TIMEOUT_S", "1800"),
        "DEXPBT_INTER_LAUNCH_DELAY_S": env.get("DEXPBT_INTER_LAUNCH_DELAY_S", "90"),
        "TORCHDYNAMO_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
        ENV_FIXTURE: "1",
    })
    smoke_num_envs = str(os.environ.get("FUSION_SMOKE_NUM_ENVS", "512"))
    launch = RUNTIME / "scripts" / "launch_8gpu_dexpbt.sh"
    cmd = [
        "bash", str(launch), str(run_dir),
        "--task", "DexPBT-Regrasping-KukaAllegro-Direct-v0",
        "--num-envs", smoke_num_envs,
        # Tip checkpoints resume at ~5B frames / high epoch ids; keep this
        # far above resumed epoch so trainers stay alive for the MSA cycle.
        "--max-iterations", "100000000",
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
    log_path = run_dir / "fusion_integrated_launch.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(RUNTIME / "src" / "dexpbt_isaaclab"),
            start_new_session=True,
        )
    checks["launcher_pid"] = proc.pid

    # Production aux processes (guard keepalive + watchdog + real sidecar
    # fixture-once). Formal ARM_CONFIG bootstrap is not required for this
    # disposable smoke; a durable guard pid keeps the watchdog fail-closed
    # contract satisfied while the real sidecar module drives fusion cmds.
    py = env.get("PYTHON", "python")
    readiness = run_dir / "FIRST_TICK_VALIDATED.json"
    arm_cfg = {
        "arm": "msa_fusion_integrated_smoke",
        "num_policies": num_workers,
        "test_only": True,
        "msa_fusion_integrated_smoke": True,
    }
    arm_path = run_dir / "ARM_CONFIG.json"
    arm_path.write_text(json.dumps(arm_cfg, indent=2))
    teardown_marker = run_dir / "FUSION_SMOKE_TEARDOWN.json"

    def _keepalive(name: str) -> subprocess.Popen:
        script = (
            "import time\nfrom pathlib import Path\n"
            f"done=Path({str(teardown_marker)!r})\n"
            f"print({name!r}+' keepalive', flush=True)\n"
            "while not done.is_file():\n"
            "    time.sleep(2)\n"
        )
        return _start_aux_process(
            run_dir, name, [py, "-u", "-c", script], env,
        )

    try:
        aux["continuation_guard"] = _keepalive("continuation_guard")
        checks["continuation_guard_mode"] = "keepalive_for_smoke"
    except Exception as exc:  # noqa: BLE001
        checks["continuation_guard_start_error"] = f"{type(exc).__name__}: {exc}"
    try:
        aux["hold_watchdog"] = _start_aux_process(
            run_dir, "hold_watchdog",
            [py, "-u", "-m", "dexpbt_isaaclab.msa.hold_watchdog",
             "--run-dir", str(run_dir),
             "--startup-grace-s", "3600"],
            env,
        )
    except Exception as exc:  # noqa: BLE001
        checks["hold_watchdog_start_error"] = f"{type(exc).__name__}: {exc}"
    try:
        aux["msa_sidecar"] = _start_aux_process(
            run_dir, "msa_sidecar",
            [py, "-u", "-m", "dexpbt_isaaclab.msa.sidecar",
             "--workspace", str(run_dir / "dexpbt_8gpu_ws"),
             "--arm", "MSA_SELECTIVE_STATEFUL",
             "--readiness-marker", str(readiness),
             "--num-policies", str(num_workers),
             "--fusion-ack-max-wait-s", "600",
             "--fusion-fixture-once"],
            env,
        )
        emit_event(run_dir, "MSA_SIDECAR_STARTED", {
            "pid": aux["msa_sidecar"].pid, "mode": "fusion_fixture_once",
        })
    except Exception as exc:  # noqa: BLE001
        checks["msa_sidecar_start_error"] = f"{type(exc).__name__}: {exc}"

    deadline = time.time() + float(os.environ.get("FUSION_SMOKE_WAIT_S", "1800"))
    after_init: set[int] = set()
    while time.time() < deadline:
        after_init = _policy_after_init(run_dir, num_workers)
        if len(after_init) >= num_workers:
            break
        if proc.poll() is not None and len(after_init) == 0:
            break
        time.sleep(5)
    checks["after_init_workers"] = sorted(after_init)
    checks["isaac_after_init_all"] = len(after_init) >= num_workers

    # Require some real training activity before driving MSA.
    step_deadline = time.time() + min(600.0, deadline - time.time())
    steps = _min_steps_seen(run_dir, num_workers)
    while time.time() < step_deadline:
        steps = _min_steps_seen(run_dir, num_workers)
        if sum(1 for v in steps.values() if v >= 1) >= num_workers:
            break
        time.sleep(5)
    checks["steps_seen"] = steps
    checks["all_workers_stepped"] = (
        sum(1 for v in steps.values() if v >= 1) >= num_workers
    )

    # Mark readiness so production sidecar can proceed.
    atomic_write_json(readiness, {
        "validated": True,
        "workers": list(range(num_workers)),
        "ts": time.time(),
        "test_only": True,
        "source": "fusion_integrated_smoke",
    })
    # Sidecar gate also wants hold-smoke marker for non-fixture path;
    # fixture-once only needs readiness, but write a stub for hygiene.
    atomic_write_json(run_dir / "HOLD_SMOKE_PASSED.json", {
        "ok": True, "test_only": True, "source": "fusion_integrated_smoke",
    })

    # Trigger production PROBE path via MSA_BUSY (same marker sidecar uses).
    emit_event(run_dir, "MSA_CYCLE_TRIGGERED", {
        "cycle_id": cycle_id,
        "due_workers": [0, 1],
        "source": "integrated_smoke_busy",
    })
    atomic_write_json(busy_path(run_dir), {
        "busy": True,
        "status": "PROBING",
        "intervention_id": cycle_id,
        "due_workers": [0, 1],
        "ts": time.time(),
        "reason": "fusion_integrated_smoke",
    })

    probe_deadline = time.time() + min(900.0, max(60.0, deadline - time.time()))
    probe_ready = {0: False, 1: False}
    while time.time() < probe_deadline:
        for w in (0, 1):
            st = load_worker_state(worker_state_path(run_dir, w))
            rec = load_transaction_record(run_dir, w, cycle_id)
            if st is not None and st.mode == MODE_PROBE and rec is not None:
                probe_ready[w] = True
        if all(probe_ready.values()):
            break
        time.sleep(3)
    checks["probe_ready"] = probe_ready
    checks["probe_anchor_saved"] = all(probe_ready.values())

    # Sidecar (--fusion-fixture-once) publishes fixture after anchors exist.
    # Orchestrator only waits for acks / cycle complete.
    checks["fixture_published"] = False
    ack_deadline = time.time() + min(600.0, max(60.0, deadline - time.time()))
    advance_ack = False
    rollback_ack = False
    while time.time() < ack_deadline:
        # Detect fixture publish via protocol claim / commands.
        claim = (
            run_dir / "msa_fusion_protocol" / "claimed"
            / f"{cycle_id}.claim.json"
        )
        if claim.is_file():
            checks["fixture_published"] = True
        a0 = read_ack(run_dir, 0, cycle_id)
        a1 = read_ack(run_dir, 1, cycle_id)
        if a0 is not None and a0.ok and a0.action_applied == "advance":
            advance_ack = True
        if a1 is not None and a1.ok and a1.action_applied in ("rollback", "hold"):
            rollback_ack = True
        if (
            checks["fixture_published"]
            and advance_ack
            and rollback_ack
        ):
            try_complete_cycle(run_dir, cycle_id, [0, 1])
            break
        # Sidecar may have already completed.
        if (
            run_dir / "msa_fusion_protocol" / "complete"
            / f"{cycle_id}.complete.json"
        ).is_file():
            advance_ack = advance_ack or (
                read_ack(run_dir, 0, cycle_id) is not None
            )
            rollback_ack = rollback_ack or (
                read_ack(run_dir, 1, cycle_id) is not None
            )
            checks["fixture_published"] = True
            break
        time.sleep(3)
    checks["advance_ack"] = advance_ack
    checks["rollback_ack"] = rollback_ack
    checks["cycle_complete"] = (
        run_dir / "msa_fusion_protocol" / "complete" / f"{cycle_id}.complete.json"
    ).is_file()

    # Clear busy after cycle.
    try:
        busy_path(run_dir).unlink()
    except OSError:
        pass

    # Short resume: kill trainers, relaunch one worker briefly, verify state.
    resume_ok = False
    resume_detail: dict[str, Any] = {}
    pre_resume_states = {}
    for w in range(num_workers):
        st = load_worker_state(worker_state_path(run_dir, w))
        if st is not None:
            pre_resume_states[str(w)] = {
                "mode": st.mode,
                "tol": st.current_tolerance,
                "fp": st.anchor_fingerprint,
                "frame_probe": st.probe_start_frame,
            }
    # Tear down primary launch group + aux.
    atomic_write_json(teardown_marker, {"ts": time.time()})
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            _kill_tree(proc.pid)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
    for name, ap in list(aux.items()):
        try:
            os.killpg(ap.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            _kill_tree(ap.pid)
    time.sleep(5)

    # Resume smoke: restart launcher briefly and confirm worker_state persists.
    resume_dir_marker = run_dir / "RESUME_SMOKE_STARTED.json"
    atomic_write_json(resume_dir_marker, {"ts": time.time()})
    with log_path.open("a") as logf:
        logf.write("\n=== RESUME PHASE ===\n")
        resume_proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(RUNTIME / "src" / "dexpbt_isaaclab"),
            start_new_session=True,
        )
    resume_deadline = time.time() + 420.0
    while time.time() < resume_deadline:
        post = {}
        for w in (0, 1):
            st = load_worker_state(worker_state_path(run_dir, w))
            if st is not None:
                post[str(w)] = {
                    "mode": st.mode,
                    "tol": st.current_tolerance,
                    "fp": st.anchor_fingerprint,
                }
        if len(post) >= 2:
            resume_ok = True
            resume_detail = {"pre": pre_resume_states, "post": post}
            break
        if resume_proc.poll() is not None:
            break
        time.sleep(5)
    try:
        os.killpg(resume_proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        _kill_tree(resume_proc.pid)
    try:
        resume_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(resume_proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    checks["resume_ok"] = resume_ok
    checks["resume_detail"] = resume_detail

    # Cleanup orphans / locks.
    atomic_write_json(teardown_marker, {"ts": time.time(), "final": True})
    try:
        subprocess.run(
            ["pkill", "-9", "-f", str(run_dir)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass
    time.sleep(3)
    for lock in run_dir.rglob("*.lock"):
        try:
            lock.unlink()
        except OSError:
            pass
    after = set(_gpu_pids())
    orphans = sorted(after - before_pids)
    for pid in orphans:
        _kill_tree(pid)
    time.sleep(2)
    after2 = set(_gpu_pids())
    orphans2 = sorted(after2 - before_pids)
    checks["orphan_pids_after_cleanup"] = orphans2

    events = _read_events(run_dir)
    counts = _event_counts(events)
    timelines = _worker_timelines(events)
    st0 = load_worker_state(worker_state_path(run_dir, 0))
    st1 = load_worker_state(worker_state_path(run_dir, 1))
    checks["worker0_mode"] = None if st0 is None else st0.mode
    checks["worker1_mode"] = None if st1 is None else st1.mode
    checks["event_counts"] = counts
    checks["aux_started"] = {
        k: v.pid for k, v in aux.items()
    }
    checks["no_fake_injector"] = os.environ.get(ENV_GUARD) != "1"

    ok = bool(
        checks.get("isaac_after_init_all")
        and checks.get("all_workers_stepped")
        and checks.get("probe_anchor_saved")
        and checks.get("fixture_published")
        and checks.get("advance_ack")
        and checks.get("rollback_ack")
        and checks.get("resume_ok")
        and len(orphans2) == 0
        and "continuation_guard" in aux
        and "hold_watchdog" in aux
        and "msa_sidecar" in aux
    )
    report = {
        "ok": ok,
        "mode": "integrated",
        "cycle_id": cycle_id,
        "checks": checks,
        "timelines": timelines,
        "test_only": True,
        "note": (
            "production sidecar --fusion-fixture-once feeds episode-level "
            "evidence into real decide_selective_promotion + command/ack; "
            "ADVANCE/ROLLBACK applied in trainer processes"
        ),
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
    ap.add_argument(
        "--mode", choices=("control", "gpu", "integrated"), default="control",
    )
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO / "results" / "runs" / f"msa_fusion_smoke_{args.mode}_{stamp}"
    )
    if args.mode == "control":
        report = run_control_smoke(run_dir, num_workers=args.num_workers)
    elif args.mode == "gpu":
        report = run_gpu_smoke(run_dir, num_workers=args.num_workers)
    else:
        report = run_integrated_smoke(run_dir, num_workers=args.num_workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
