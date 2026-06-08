#!/usr/bin/env python3
"""Launch N parallel DexPBT learners with a shared workspace.

This launcher supports staggered policy launches with per-policy Isaac Sim
isolation and a startup watchdog. Key design points:

* Warming pool: ``--max-active`` caps the number of policies that have NOT
  yet reached "first epoch". Once a policy emits its first-epoch marker the
  slot is freed and the next policy can start.
* Inter-launch delay: ``--inter-launch-delay-s`` enforces a minimum wall-clock
  spacing between successive Popen calls (default 90s) so Isaac Sim's
  Omniverse extension cache and CUDA initialisation do not collide.
* Per-policy XDG isolation: each policy gets its own ``XDG_DATA_HOME`` /
  ``XDG_CACHE_HOME`` under ``RUN_DIR/policy_NN/.xdg_*``. Read-only Omniverse
  extension dirs are symlinked from the shared
  ``runtime_root/cache/ov/data/exts/v2`` so disk cost stays minimal while
  ``omni.kvdb`` / ``registry.lock`` files stay private per process.
* Startup watchdog: a background thread tails ``policy_NN/train.log`` and
  looks for any of the configured markers. If none appears within
  ``--startup-timeout-s`` it writes a ``startup_hang.flag`` file, SIGTERMs the
  ``train.py`` PID (read from ``train_pid.txt``) and lets the supervisor's
  retry loop spawn attempt N+1. After ``--startup-max-restarts`` failed
  attempts the watchdog records ``startup_hang_failed`` and stops; the
  supervisor (which honours ``STARTUP_MAX_RESTARTS``) marks the policy
  ``status=stopped reason=startup_hang exit_code=0`` so the active slot is
  released.

All launch-side events are appended to ``RUN_DIR/launch_audit.jsonl`` for the
P2 smoke report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SUPERVISOR = Path(__file__).resolve().parents[2] / "run_train_supervisor.sh"

DEFAULT_FIRST_EPOCH_MARKERS = (
    "epoch: 1/",
    "fps step:",
    "PPO total step",
)


def _read_status(run_dir: Path) -> dict[str, str]:
    status_path = run_dir / "status.txt"
    if not status_path.is_file():
        return {}
    status: dict[str, str] = {}
    for line in status_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            status[key] = value
    return status


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_is_active(run_dir: Path) -> bool:
    pid_path = run_dir / "pid.txt"
    if not pid_path.is_file():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return False
    return _pid_is_alive(pid)


def _run_is_complete(run_dir: Path) -> bool:
    status = _read_status(run_dir)
    return status.get("status") == "stopped" and status.get("exit_code") == "0"


def _active_supervisor_count(root: Path, ignore_pids: set[int] | None = None) -> int:
    ignore_pids = ignore_pids or set()
    count = 0
    for pid_path in root.glob("**/pid.txt"):
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        if _pid_is_alive(pid):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per-policy Isaac Sim cache isolation
# ---------------------------------------------------------------------------


def _runtime_root_from_env() -> Path:
    raw = os.environ.get("RUNTIME_ROOT", "/root/autodl-tmp/wmdexpbt/runtime_root")
    return Path(raw).resolve()


def _clean_policy_locks(policy_dir: Path) -> None:
    """Wipe stale Omniverse / kvdb lock files before each spawn.

    Locks accumulate when an Isaac Sim process exits ungracefully, and the
    next start-up refuses to take ownership of the kvdb. Also removes any
    lingering startup_hang.flag from a prior watchdog event.
    """

    patterns = (
        ".xdg_data/**/*.lock",
        ".xdg_data/**/registry.lock",
        ".xdg_data/**/_lock",
        ".xdg_data/**/kvdb*.lock",
        ".xdg_cache/**/*.lock",
        ".xdg_cache/**/_lock",
    )
    for pat in patterns:
        for f in policy_dir.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass

    flag = policy_dir / "startup_hang.flag"
    try:
        flag.unlink()
    except FileNotFoundError:
        pass

    train_pid = policy_dir / "train_pid.txt"
    try:
        train_pid.unlink()
    except FileNotFoundError:
        pass


def _prepare_policy_ov_root(policy_dir: Path, runtime_root: Path) -> tuple[Path, Path]:
    """Build per-policy XDG_DATA_HOME / XDG_CACHE_HOME for Omniverse.

    Layout::

        policy_NN/.xdg_data/ov/data/exts/v2/
            cache_db.json   (best-effort copy from shared)
            index/          (per-policy mkdir, writable)
            omni.*/         (symlinks -> shared exts/v2/omni.*/)
        policy_NN/.xdg_cache/ov/client/
            _cache.json     (best-effort copy from shared)

    Returns ``(xdg_data_home, xdg_cache_home)``.
    """

    xdg_data_home = policy_dir / ".xdg_data"
    xdg_cache_home = policy_dir / ".xdg_cache"

    per_v2 = xdg_data_home / "ov" / "data" / "exts" / "v2"
    per_v2.mkdir(parents=True, exist_ok=True)
    (per_v2 / "index").mkdir(exist_ok=True)

    per_client = xdg_cache_home / "ov" / "client"
    per_client.mkdir(parents=True, exist_ok=True)

    shared_v2 = runtime_root / "cache" / "ov" / "data" / "exts" / "v2"
    shared_client = runtime_root / "cache" / "ov_client" / "client"

    # Symlink each shared extension dir into the per-policy v2 root.
    if shared_v2.is_dir():
        for entry in shared_v2.iterdir():
            if entry.name in ("index", "cache_db.json"):
                continue
            target = per_v2 / entry.name
            if target.exists() or target.is_symlink():
                continue
            try:
                target.symlink_to(entry.resolve())
            except OSError as exc:
                print(
                    f"[isolate] WARN: failed to symlink {entry} -> {target}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        print(
            f"[isolate] WARN: shared exts/v2 not found at {shared_v2}; "
            "Isaac Sim will re-download extensions on first launch.",
            file=sys.stderr,
            flush=True,
        )

    # Best-effort cache_db.json / _cache.json copies. Missing source -> warn.
    src_cache_db = shared_v2 / "cache_db.json"
    dst_cache_db = per_v2 / "cache_db.json"
    if not dst_cache_db.exists():
        if src_cache_db.is_file():
            try:
                shutil.copy2(src_cache_db, dst_cache_db)
            except OSError as exc:
                print(
                    f"[isolate] WARN: copy {src_cache_db} -> {dst_cache_db} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            print(
                f"[isolate] WARN: shared cache_db.json missing at {src_cache_db}; "
                "starting with empty per-policy cache.",
                file=sys.stderr,
                flush=True,
            )

    src_client_cache = shared_client / "_cache.json"
    dst_client_cache = per_client / "_cache.json"
    if not dst_client_cache.exists():
        if src_client_cache.is_file():
            try:
                shutil.copy2(src_client_cache, dst_client_cache)
            except OSError as exc:
                print(
                    f"[isolate] WARN: copy {src_client_cache} -> {dst_client_cache} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            print(
                f"[isolate] WARN: shared _cache.json missing at {src_client_cache}; "
                "starting with empty per-policy client cache.",
                file=sys.stderr,
                flush=True,
            )

    return xdg_data_home.resolve(), xdg_cache_home.resolve()


# ---------------------------------------------------------------------------
# Launch audit log
# ---------------------------------------------------------------------------


_AUDIT_LOCK = threading.Lock()


def _audit(run_root: Path, policy_idx: int, event: str, **fields) -> None:
    payload = {
        "policy_idx": int(policy_idx),
        "event": str(event),
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(fields)
    line = json.dumps(payload, ensure_ascii=False)
    audit_path = run_root / "launch_audit.jsonl"
    with _AUDIT_LOCK:
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Startup watchdog
# ---------------------------------------------------------------------------


class _StartupWatchdog(threading.Thread):
    """Per-policy thread that tails train.log and enforces a startup timeout.

    The watchdog stops once the supervisor process exits (its job is launch
    only, not lifetime).
    """

    def __init__(
        self,
        run_root: Path,
        policy_dir: Path,
        policy_idx: int,
        markers: tuple[str, ...],
        timeout_s: float,
        max_restarts: int,
        proc: subprocess.Popen,
        warmed_set: set[int],
        warmed_lock: threading.Lock,
    ) -> None:
        super().__init__(daemon=True, name=f"watchdog-policy-{policy_idx:02d}")
        self.run_root = run_root
        self.policy_dir = policy_dir
        self.policy_idx = int(policy_idx)
        self.markers = tuple(markers)
        self.timeout_s = float(timeout_s)
        self.max_restarts = int(max_restarts)
        self.proc = proc
        self.warmed = warmed_set
        self.warmed_lock = warmed_lock
        self.train_log = policy_dir / "train.log"
        self.flag_path = policy_dir / "startup_hang.flag"
        self.train_pid_path = policy_dir / "train_pid.txt"
        self._poll_interval = 2.0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _read_train_pid(self) -> int | None:
        if not self.train_pid_path.is_file():
            return None
        try:
            return int(self.train_pid_path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _scan_for_marker(self, fh) -> bool:
        # Read whatever is available; do not block.
        chunk = fh.read()
        if not chunk:
            return False
        for marker in self.markers:
            if marker in chunk:
                with self.warmed_lock:
                    self.warmed.add(self.policy_idx)
                _audit(
                    self.run_root,
                    self.policy_idx,
                    "policy_first_epoch_time",
                    matched_marker=marker,
                )
                return True
        return False

    def _wait_for_log(self, deadline: float) -> object:
        """Block until train.log exists or supervisor exits."""

        while time.monotonic() < deadline:
            if self.train_log.is_file():
                return self.train_log.open("r", encoding="utf-8", errors="replace")
            if self.proc.poll() is not None:
                return None
            if self._stop.is_set():
                return None
            time.sleep(0.5)
        return None

    def _trigger_hang(self, attempt: int) -> None:
        train_pid = self._read_train_pid()
        flag_payload = {
            "reason": "startup_hang",
            "timeout_s": self.timeout_s,
            "attempt": int(attempt),
            "ts": time.time(),
            "train_pid": train_pid,
        }
        try:
            self.flag_path.write_text(json.dumps(flag_payload))
        except OSError as exc:
            print(
                f"[watchdog] WARN: cannot write {self.flag_path}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        _audit(
            self.run_root,
            self.policy_idx,
            "startup_timeout",
            action="sigterm_train",
            attempt=int(attempt),
            train_pid=train_pid,
        )
        if train_pid is not None and _pid_is_alive(train_pid):
            try:
                os.kill(train_pid, signal.SIGTERM)
            except OSError as exc:
                print(
                    f"[watchdog] WARN: SIGTERM train_pid={train_pid} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def run(self) -> None:  # noqa: C901 - linear FSM
        attempt = 1
        hang_count = 0
        while not self._stop.is_set():
            if self.proc.poll() is not None:
                return  # supervisor exited
            log_fh = self._wait_for_log(time.monotonic() + 30.0)
            if log_fh is None:
                # supervisor exited or stop requested before log appeared
                continue
            log_fh.seek(0)
            attempt_deadline = time.monotonic() + self.timeout_s
            warmed_now = False
            while not self._stop.is_set():
                if self.proc.poll() is not None:
                    log_fh.close()
                    return
                if self._scan_for_marker(log_fh):
                    warmed_now = True
                    break
                if time.monotonic() >= attempt_deadline:
                    break
                time.sleep(self._poll_interval)
            log_fh.close()
            if warmed_now:
                return
            # Timeout in this attempt.
            hang_count += 1
            self._trigger_hang(attempt)
            if hang_count > self.max_restarts:
                _audit(
                    self.run_root,
                    self.policy_idx,
                    "startup_hang_failed",
                    attempt=int(attempt),
                    hang_count=int(hang_count),
                )
                return
            _audit(
                self.run_root,
                self.policy_idx,
                "restart_due_to_startup_hang",
                attempt=int(attempt + 1),
                hang_count=int(hang_count),
            )
            # Wait briefly for supervisor to spawn the next attempt; the
            # supervisor will rewrite train_pid.txt and append a new
            # "========== attempt_X" separator to train.log.
            time.sleep(5.0)
            attempt += 1


# ---------------------------------------------------------------------------
# Concurrency loop
# ---------------------------------------------------------------------------


PolicyTuple = tuple[subprocess.Popen, str, object, _StartupWatchdog | None, int]


def _poll_running(running: list[PolicyTuple]) -> list[PolicyTuple]:
    still_running: list[PolicyTuple] = []
    for proc, label, log_file, watchdog, policy_idx in running:
        exit_code = proc.poll()
        if exit_code is None:
            still_running.append((proc, label, log_file, watchdog, policy_idx))
            continue
        log_file.close()
        if watchdog is not None:
            watchdog.stop()
        if exit_code != 0:
            raise RuntimeError(f"{label} failed with exit code {exit_code}")
    return still_running


def _warming_count(running: list[PolicyTuple], warmed: set[int]) -> int:
    return sum(1 for _, _, _, _, pidx in running if pidx not in warmed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DexPBT multi-policy training.")
    parser.add_argument("--num-policies", type=int, default=6)
    parser.add_argument("--task", default="DexPBT-Reorientation-KukaAllegro-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=38147)
    parser.add_argument("--train-root", type=str, required=True)
    parser.add_argument("--workspace", type=str, default="dexpbt_single_arm_ws")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1")
    parser.add_argument("--max-active", type=int, default=None, help="Max policies in 'warming' state")
    parser.add_argument(
        "--inter-launch-delay-s",
        type=float,
        default=90.0,
        help="Minimum seconds between successive policy spawns",
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=900.0,
        help="Per-attempt timeout to reach the first-epoch marker",
    )
    parser.add_argument(
        "--first-epoch-marker",
        action="append",
        default=None,
        help="Substring in train.log that signals first epoch reached. "
        "May be passed multiple times; defaults to a curated list if unset.",
    )
    parser.add_argument(
        "--startup-max-restarts",
        type=int,
        default=1,
        help="Number of startup-hang retries before giving up on a policy",
    )
    parser.add_argument("--pbt-interval-steps", type=int, default=20_000_000)
    parser.add_argument("--pbt-start-after", type=int, default=200_000_000)
    parser.add_argument("--pbt-initial-delay", type=int, default=50_000_000)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--change-min", type=float, default=1.1)
    parser.add_argument("--change-max", type=float, default=1.5)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra CLI overrides passed through to train.py")
    parser.add_argument(
        "--recover-checkpoint",
        type=str,
        default=None,
        help="Optional PBT workspace checkpoint for warm-starting one policy.",
    )
    parser.add_argument(
        "--recover-policy-idx",
        type=int,
        default=3,
        help="Policy index that receives --recover-checkpoint on first launch.",
    )
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    train_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided via --gpus")
    max_active = args.max_active if args.max_active is not None else len(gpu_ids)

    markers = tuple(args.first_epoch_marker) if args.first_epoch_marker else DEFAULT_FIRST_EPOCH_MARKERS

    runtime_root = _runtime_root_from_env()

    # All policies share a single workspace directory so they can see each
    # other's objectives.  Place it at the top of train_root.
    shared_workspace = (train_root / args.workspace).resolve()
    shared_workspace.mkdir(parents=True, exist_ok=True)

    print(
        f"[launch_pbt] num_policies={args.num_policies} max_active={max_active} "
        f"inter_launch_delay_s={args.inter_launch_delay_s} "
        f"startup_timeout_s={args.startup_timeout_s} "
        f"startup_max_restarts={args.startup_max_restarts} "
        f"first_epoch_markers={list(markers)}",
        flush=True,
    )

    warmed: set[int] = set()
    warmed_lock = threading.Lock()
    last_launch_ts = 0.0
    running: list[PolicyTuple] = []
    for policy_idx in range(args.num_policies):
        run_dir = train_root / f"policy_{policy_idx:02d}"
        if _run_is_complete(run_dir):
            print(f"Skip completed policy {policy_idx}", flush=True)
            with warmed_lock:
                warmed.add(policy_idx)
            continue
        if _run_is_active(run_dir):
            print(f"Skip active policy {policy_idx}", flush=True)
            with warmed_lock:
                warmed.add(policy_idx)
            continue

        # Concurrency + delay gate.
        while True:
            running = _poll_running(running)
            running_pids = {proc.pid for proc, _, _, _, _ in running}
            active_count = _active_supervisor_count(train_root, ignore_pids=running_pids)
            warming = _warming_count(running, warmed) + active_count
            elapsed = time.monotonic() - last_launch_ts
            if warming < max_active and elapsed >= args.inter_launch_delay_s:
                break
            time.sleep(2.0)

        gpu = gpu_ids[policy_idx % len(gpu_ids)]
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        # Per-policy isolation: clean stale locks, materialise XDG dirs.
        _clean_policy_locks(run_dir)
        xdg_data, xdg_cache = _prepare_policy_ov_root(run_dir, runtime_root)

        cmd = [
            "bash",
            str(SUPERVISOR),
            str(run_dir),
            args.task,
            "--device=cuda:0",
            "--headless",
            "--observer_mode=pbt",
            f"--policy_idx={policy_idx}",
            f"--num_policies={args.num_policies}",
            f"--workspace={str(shared_workspace)}",
            f"--pbt_interval_steps={args.pbt_interval_steps}",
            f"--pbt_start_after={args.pbt_start_after}",
            f"--pbt_initial_delay={args.pbt_initial_delay}",
            f"--pbt_mutation_rate={args.mutation_rate}",
            f"--pbt_change_min={args.change_min}",
            f"--pbt_change_max={args.change_max}",
            f"--num_envs={args.num_envs}",
            f"--max_iterations={args.max_iterations}",
        ]
        # Normalize WM flags to --key=value so PBT restart argv reconstruction
        # does not leave orphan tokens (value_imag_h1, 3, 5, ...).
        boolean_flags = {
            "--enable_wm",
            "--enable_counterfactual",
            "--cf_reset_epoch_counter",
            "--cf_require_nsucc_gt_zero",
            "--no-cf_require_nsucc_gt_zero",
            "--require_counterfactual_runner",
            # Patch A (rev 3, 2026-05-30): legacy source fallback toggle.
            "--pbt_allow_low_nsucc_source_fallback",
            "--no-pbt_allow_low_nsucc_source_fallback",
            # Patch F (2026-05-31): self-restart governor toggles.
            "--pbt_disable_self_restart",
            "--no-pbt_disable_self_restart",
            "--pbt_allow_mid_self_mutation",
            "--no-pbt_allow_mid_self_mutation",
            # Patch G (2026-05-31): best_nsucc recovery toggle.
            "--pbt_enable_best_nsucc_recovery",
            "--no-pbt_enable_best_nsucc_recovery",
            # Patch P1 (2026-06-01): mastery curriculum + stable-source toggles.
            "--pbt_source_require_stable_mean",
            "--no-pbt_source_require_stable_mean",
            "--curriculum_collapse_detect",
            "--no-curriculum_collapse_detect",
            "--pbt_allow_best_nsucc_archive_recovery",
            "--no-pbt_allow_best_nsucc_archive_recovery",
            # Patch P2-A (2026-06-01): single-shot shrink token.
            "--pbt_single_shrink_per_window",
            "--no-pbt_single_shrink_per_window",
            "--pbt_early_collapse_detect",
            "--no-pbt_early_collapse_detect",
            "--pbt_recovery_bypass_elimination_burnin",
            "--no-pbt_recovery_bypass_elimination_burnin",
            # Phase A4 sequence WM toggles.
            "--seq_wm_value_detach_belief",
            "--no-seq_wm_value_detach_belief",
            # Phase A4 P0: PPO-facing distillation has only value flags
            # (--seq_wm_policy_coupling=...) so no boolean entries here;
            # the generic `--key value` -> `--key=value` normalisation
            # below handles them automatically.
        }
        extra = list(args.extra)
        i = 0
        normalized: list[str] = []
        while i < len(extra):
            tok = extra[i]
            if tok == "--extra":
                i += 1
                continue
            if tok in boolean_flags:
                normalized.append(tok)
                i += 1
                continue
            if tok.startswith("--") and "=" not in tok and i + 1 < len(extra):
                normalized.append(f"{tok}={extra[i + 1]}")
                i += 2
                continue
            normalized.append(tok)
            i += 1
        cmd.extend(normalized)
        if (
            args.recover_checkpoint
            and policy_idx == args.recover_policy_idx
            and not _run_is_complete(run_dir)
            and not _run_is_active(run_dir)
        ):
            cmd.append(f"--checkpoint={args.recover_checkpoint}")
        print(f"Launch policy {policy_idx}: {' '.join(cmd)}", flush=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["XDG_DATA_HOME"] = str(xdg_data)
        env["XDG_CACHE_HOME"] = str(xdg_cache)
        env["STARTUP_MAX_RESTARTS"] = str(int(args.startup_max_restarts) + 1)
        env["DEXPBT_STARTUP_HANG_FLAG"] = str(run_dir / "startup_hang.flag")
        env["DEXPBT_FIRST_EPOCH_MARKERS"] = "||".join(markers)
        env["DEXPBT_TRAIN_PID_FILE"] = str(run_dir / "train_pid.txt")

        log_file = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        last_launch_ts = time.monotonic()
        _audit(
            train_root,
            policy_idx,
            "policy_start_time",
            gpu=gpu,
            attempt=1,
            cmd=cmd,
            xdg_data_home=str(xdg_data),
            xdg_cache_home=str(xdg_cache),
        )
        watchdog = _StartupWatchdog(
            run_root=train_root,
            policy_dir=run_dir,
            policy_idx=policy_idx,
            markers=markers,
            timeout_s=args.startup_timeout_s,
            max_restarts=args.startup_max_restarts,
            proc=proc,
            warmed_set=warmed,
            warmed_lock=warmed_lock,
        )
        watchdog.start()
        running.append((proc, f"DexPBT policy={policy_idx}", log_file, watchdog, policy_idx))

    while running:
        running = _poll_running(running)
        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
