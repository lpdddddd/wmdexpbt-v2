#!/usr/bin/env python3
"""Checkpoint manifest builder for CCRS diagnostics.

Reads a finished (or near-finished) PBT run workspace and produces a manifest
of checkpoints relevant to the CCRS sweep, without modifying anything in the
run directory (strictly read-only).

Each manifest row records (CSV columns):
  run_id, git_commit, worker_id, iteration, frame, checkpoint_path,
  live_tolerance, live_nsucc, historical_peak_nsucc, lineage_id,
  policy_arch, roles, target_frame, frame_deviation,
  file_size_bytes, zip_ok, partial_sha256, torch_load_ok (optional)

Tolerance units are meters throughout.

CLI:
  python -m dexpbt_isaaclab.ccrs.manifest \
    --run-dir results/runs/<run> \
    --out-dir results/analysis/ccrs_base_retention \
    [--targets '1=4.12e9,4=3.48e9,5=2.52e9,3=4.92e9,2=1.92e9'] \
    [--transition-band 0.018:0.0285] \
    [--torch-check none|selected|all]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

TOL_KEY = "success_tolerance"

# Frames closer than this to a requested target count as "exact".
EXACT_FRAME_TOL = 10_000_000  # 10M frames


def _read_meta(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _meta_tol(meta: dict) -> Optional[float]:
    curr = meta.get("curriculum")
    if isinstance(curr, dict) and curr.get(TOL_KEY) is not None:
        return float(curr[TOL_KEY])
    return None


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _partial_sha256(path: Path, head_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash of (file size || first head_bytes). Cheap identity for large ckpts."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def _zip_ok(path: Path) -> bool:
    """Torch >=1.6 checkpoints are zip archives; validate structure cheaply."""
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            return any(n.endswith("data.pkl") for n in names)
    except Exception:
        return False


def _torch_load_ok(path: Path) -> bool:
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state = ckpt.get("model") if isinstance(ckpt, dict) else None
        return isinstance(state, dict) and len(state) > 0
    except Exception:
        return False


@dataclass
class ManifestRow:
    run_id: str
    # Commit of the CODE at manifest-generation time (evaluation toolkit),
    # NOT the commit the checkpoints were trained with — see training_commit.
    git_commit: str
    # Commit the training run was launched from (provenance of checkpoints).
    training_commit: str
    worker_id: int
    iteration: int
    frame: int
    checkpoint_path: str
    live_tolerance: Optional[float]
    live_nsucc: Optional[float]
    historical_peak_nsucc: Optional[float]
    lineage_id: Optional[str]
    policy_arch: str
    roles: str = ""            # semicolon-joined role tags
    target_frame: Optional[int] = None
    frame_deviation: Optional[int] = None
    file_size_bytes: int = -1
    zip_ok: bool = False
    partial_sha256: str = ""
    torch_load_ok: Optional[bool] = None
    notes: str = ""


@dataclass
class TransitionPair:
    """A curriculum shrink transition: checkpoint before (p) and after (q).

    lineage_changed=True marks a lineage_replacement_transition: the post
    checkpoint carries a different lineage_id (PBT copy/replacement between
    the two ticks). Its F component mixes replacement effects and must not
    be interpreted as ordinary within-lineage policy update / forgetting.
    """
    worker_id: int
    pre_iteration: int
    post_iteration: int
    pre_frame: int
    post_frame: int
    pre_tolerance: float
    post_tolerance: float
    pre_nsucc: Optional[float] = None
    post_nsucc: Optional[float] = None
    pre_lineage_id: Optional[str] = None
    post_lineage_id: Optional[str] = None
    lineage_changed: Optional[bool] = None


def parse_targets(spec: Optional[str]) -> dict[int, list[float]]:
    """Parse '1=4.12e9,4=3.48e9,5=2.52e9|1.0e9' into {worker: [frames...]}.

    Multiple targets for the same worker may be given either by repeating
    the key or with '|' separators inside the value.
    """
    out: dict[int, list[float]] = {}
    if not spec:
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"invalid target entry {tok!r}; expected worker=frames")
        k, v = tok.split("=", 1)
        vals = [float(x) for x in v.split("|") if x.strip()]
        out.setdefault(int(k.strip()), []).extend(vals)
    return out


def load_worker_metas(workspace: Path, worker_id: int) -> list[dict]:
    """All parseable PBT tick metas for one worker, sorted by frame."""
    d = workspace / f"{worker_id:03d}"
    metas = []
    if not d.is_dir():
        return metas
    for p in sorted(d.glob("*.json")):
        m = _read_meta(p)
        if m is None or "frame" not in m:
            continue
        m["_meta_path"] = str(p)
        metas.append(m)
    metas.sort(key=lambda m: (int(m["frame"]), int(m.get("iteration", -1))))
    return metas


def find_transitions(
    metas: list[dict],
    band_lo: float,
    band_hi: float,
) -> list[TransitionPair]:
    """Detect tolerance-shrink transitions whose *post* tolerance is in band."""
    out: list[TransitionPair] = []
    prev = None
    for m in metas:
        tol = _meta_tol(m)
        if tol is None:
            continue
        if prev is not None:
            ptol = _meta_tol(prev)
            if ptol is not None and tol < ptol - 1e-9 and band_lo <= tol <= band_hi:
                pre_lin = prev.get("lineage_id")
                post_lin = m.get("lineage_id")
                out.append(TransitionPair(
                    worker_id=int(m.get("policy_idx", -1)),
                    pre_iteration=int(prev.get("iteration", -1)),
                    post_iteration=int(m.get("iteration", -1)),
                    pre_frame=int(prev["frame"]),
                    post_frame=int(m["frame"]),
                    pre_tolerance=float(ptol),
                    post_tolerance=float(tol),
                    pre_nsucc=float(prev["nsucc_mean"]) if prev.get("nsucc_mean") is not None else None,
                    post_nsucc=float(m["nsucc_mean"]) if m.get("nsucc_mean") is not None else None,
                    pre_lineage_id=pre_lin,
                    post_lineage_id=post_lin,
                    lineage_changed=(None if (pre_lin is None or post_lin is None)
                                     else bool(pre_lin != post_lin)),
                ))
        prev = m
    return out


def nearest_meta(metas: list[dict], target_frame: float) -> Optional[dict]:
    if not metas:
        return None
    return min(metas, key=lambda m: abs(int(m["frame"]) - target_frame))


def build_manifest(
    run_dir: Path,
    repo_root: Path,
    targets: dict[int, list[float]],
    band_lo: float = 0.018,
    band_hi: float = 0.0285,
    torch_check: str = "none",
    workers: Optional[list[int]] = None,
    policy_arch: str = "rl_games a2c_continuous LSTM (DexPBTKukaAllegroLSTMPPO)",
    training_commit: str = "unknown",
) -> tuple[list[ManifestRow], list[TransitionPair]]:
    """Build manifest rows + transition list. Read-only.

    git_commit records the CURRENT code commit (evaluation toolkit);
    training_commit must be supplied by the caller and records the commit
    the run's checkpoints were actually trained with.
    """
    workspace = run_dir / "dexpbt_8gpu_ws"
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace not found: {workspace}")
    run_id = run_dir.name
    commit = _git_commit(repo_root)
    if workers is None:
        workers = sorted(
            int(p.name) for p in workspace.iterdir()
            if p.is_dir() and p.name.isdigit()
        )

    rows: list[ManifestRow] = []
    transitions: list[TransitionPair] = []
    # (worker, iteration) -> row (avoid duplicates across role sources)
    index: dict[tuple[int, int], ManifestRow] = {}

    def add_row(worker_id: int, meta: dict, role: str,
                target_frame: Optional[float] = None) -> Optional[ManifestRow]:
        it = int(meta.get("iteration", -1))
        key = (worker_id, it)
        if key in index:
            row = index[key]
            if role not in row.roles.split(";"):
                row.roles = f"{row.roles};{role}" if row.roles else role
            if target_frame is not None and row.target_frame is None:
                row.target_frame = int(target_frame)
                row.frame_deviation = int(abs(int(meta["frame"]) - target_frame))
            return row
        ckpt_str = meta.get("checkpoint", "")
        ckpt = Path(ckpt_str) if ckpt_str else None
        exists = bool(ckpt and ckpt.is_file())
        peak = None
        for m in metas_by_worker[worker_id]:
            if int(m["frame"]) <= int(meta["frame"]) and m.get("nsucc_mean") is not None:
                v = float(m["nsucc_mean"])
                peak = v if peak is None else max(peak, v)
        row = ManifestRow(
            run_id=run_id,
            git_commit=commit,
            training_commit=training_commit,
            worker_id=worker_id,
            iteration=it,
            frame=int(meta["frame"]),
            checkpoint_path=str(ckpt) if ckpt else "",
            live_tolerance=_meta_tol(meta),
            live_nsucc=float(meta["nsucc_mean"]) if meta.get("nsucc_mean") is not None else None,
            historical_peak_nsucc=peak,
            lineage_id=meta.get("lineage_id"),
            policy_arch=policy_arch,
            roles=role,
            target_frame=int(target_frame) if target_frame is not None else None,
            frame_deviation=(int(abs(int(meta["frame"]) - target_frame))
                             if target_frame is not None else None),
            file_size_bytes=ckpt.stat().st_size if exists else -1,
            zip_ok=_zip_ok(ckpt) if exists else False,
            partial_sha256=_partial_sha256(ckpt) if exists else "",
            notes="" if exists else "CHECKPOINT_MISSING",
        )
        index[key] = row
        rows.append(row)
        return row

    metas_by_worker = {w: load_worker_metas(workspace, w) for w in workers}

    for w in workers:
        metas = metas_by_worker[w]
        if not metas:
            continue
        # tip = last tick
        add_row(w, metas[-1], "tip")
        # global peak by nsucc
        with_ns = [m for m in metas if m.get("nsucc_mean") is not None]
        if with_ns:
            peak_meta = max(with_ns, key=lambda m: float(m["nsucc_mean"]))
            add_row(w, peak_meta, "peak")
        # requested targets
        for tgt in targets.get(w, []):
            m = nearest_meta(metas, tgt)
            if m is None:
                continue
            role = "target" if abs(int(m["frame"]) - tgt) <= EXACT_FRAME_TOL else "target_nearest"
            add_row(w, m, role, target_frame=tgt)
        # tolerance shrink transitions in band
        trs = find_transitions(metas, band_lo, band_hi)
        transitions.extend(trs)
        by_iter = {int(m.get("iteration", -1)): m for m in metas}
        for tr in trs:
            pre = by_iter.get(tr.pre_iteration)
            post = by_iter.get(tr.post_iteration)
            if pre is not None:
                add_row(w, pre, f"transition_pre@{tr.post_tolerance:.4f}")
            if post is not None:
                add_row(w, post, f"transition_post@{tr.post_tolerance:.4f}")

    # optional torch load validation
    if torch_check != "none":
        selected = rows if torch_check == "all" else [
            r for r in rows
            if any(t in r.roles for t in ("tip", "peak", "target"))
        ]
        for r in selected:
            if r.checkpoint_path and Path(r.checkpoint_path).is_file():
                r.torch_load_ok = _torch_load_ok(Path(r.checkpoint_path))

    rows.sort(key=lambda r: (r.worker_id, r.frame))
    return rows, transitions


def write_outputs(
    rows: list[ManifestRow],
    transitions: list[TransitionPair],
    out_dir: Path,
    run_dir: Path,
    extra_notes: Optional[list[str]] = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "checkpoint_manifest.csv"
    md_path = out_dir / "CHECKPOINT_AUDIT.md"

    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        f.name for f in ManifestRow.__dataclass_fields__.values()
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    trans_path = out_dir / "tolerance_transitions.csv"
    tfields = [f for f in TransitionPair.__dataclass_fields__]
    with trans_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields)
        w.writeheader()
        for t in transitions:
            w.writerow(asdict(t))

    missing = [r for r in rows if "CHECKPOINT_MISSING" in r.notes]
    bad_zip = [r for r in rows if r.file_size_bytes > 0 and not r.zip_ok]
    lines = [
        "# CCRS Checkpoint Audit",
        "",
        f"- run: `{run_dir}`",
        (f"- eval code commit (this audit): `{rows[0].git_commit}`  |  "
         f"training commit (checkpoints): `{rows[0].training_commit}`"
         if rows else ""),
        f"- rows: {len(rows)}  |  transitions in band: {len(transitions)}",
        f"- missing checkpoints: {len(missing)}",
        f"- zip-invalid checkpoints: {len(bad_zip)}",
        "",
        "Tolerance unit: meters (m); 0.020 m = 2.0 cm.",
        "",
        "## Selected checkpoints",
        "",
        "| worker | iter | frame | tol(m) | nsucc | peak_nsucc | roles | ok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ok = "yes" if (r.zip_ok and not r.notes) else ("MISSING" if r.notes else "zip_bad")
        tol = f"{r.live_tolerance:.4f}" if r.live_tolerance is not None else "-"
        ns = f"{r.live_nsucc:.1f}" if r.live_nsucc is not None else "-"
        pk = f"{r.historical_peak_nsucc:.1f}" if r.historical_peak_nsucc is not None else "-"
        lines.append(
            f"| p{r.worker_id} | {r.iteration} | {r.frame/1e9:.2f}B | {tol} | {ns} | {pk} | {r.roles} | {ok} |"
        )
    lines += [
        "",
        "## Tolerance shrink transitions (post-tol in band)",
        "",
        "lineage=changed marks a lineage_replacement_transition (PBT copy "
        "between ticks); excluded from within-lineage forgetting analysis.",
        "",
        "| worker | pre_iter -> post_iter | frames | tol(m) | nsucc | lineage |",
        "|---|---|---|---|---|---|",
    ]
    for t in transitions:
        lin = ("changed" if t.lineage_changed
               else ("same" if t.lineage_changed is not None else "unknown"))
        lines.append(
            f"| p{t.worker_id} | {t.pre_iteration} -> {t.post_iteration} "
            f"| {t.pre_frame/1e9:.2f}->{t.post_frame/1e9:.2f}B "
            f"| {t.pre_tolerance:.4f}->{t.post_tolerance:.4f} "
            f"| {t.pre_nsucc:.1f}->{t.post_nsucc:.1f} | {lin} |"
        )
    if extra_notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in extra_notes]
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CCRS checkpoint manifest (read-only).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-root", default=None,
                    help="Repo root for git SHA (default: auto from run-dir parents)")
    ap.add_argument("--targets", default="1=4.12e9,4=3.48e9,5=2.52e9,3=4.92e9,2=1.92e9|2.2e9")
    ap.add_argument("--transition-band", default="0.018:0.0285",
                    help="post-shrink tolerance band lo:hi in meters")
    ap.add_argument("--torch-check", choices=["none", "selected", "all"], default="selected")
    ap.add_argument("--note", action="append", default=[],
                    help="Extra note lines for the audit MD (repeatable)")
    ap.add_argument("--training-commit", default="unknown",
                    help="Git commit the run's checkpoints were trained with "
                         "(provenance; distinct from the current code commit)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else run_dir.parents[2]
    lo, hi = (float(x) for x in args.transition_band.split(":"))
    rows, transitions = build_manifest(
        run_dir=run_dir,
        repo_root=repo_root,
        targets=parse_targets(args.targets),
        band_lo=lo,
        band_hi=hi,
        torch_check=args.torch_check,
        training_commit=args.training_commit,
    )
    csv_path, md_path = write_outputs(
        rows, transitions, Path(args.out_dir).resolve(), run_dir,
        extra_notes=args.note,
    )
    print(f"[ccrs.manifest] wrote {len(rows)} rows -> {csv_path}")
    print(f"[ccrs.manifest] audit -> {md_path}")


if __name__ == "__main__":
    main()
