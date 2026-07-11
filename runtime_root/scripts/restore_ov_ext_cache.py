#!/usr/bin/env python3
"""Restore broken Kit extension symlinks by downloading from the local registry index."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RUNTIME_ROOT = Path(os.environ.get("RUNTIME_ROOT", "/root/autodl-tmp/wmdexpbt/runtime_root"))
CACHE = RUNTIME_ROOT / "cache/ov/data/exts/v2"
EXTS3 = (
    RUNTIME_ROOT
    / "envs/isaaclab311/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3"
)
ZIP_DIR = Path("/tmp/ov_ext_zips")
WORKERS = int(os.environ.get("OV_EXT_WORKERS", "2"))
MAX_DOWNLOAD_RETRIES = int(os.environ.get("OV_EXT_MAX_RETRIES", "5"))

db_lock = threading.Lock()

import urllib.request

def expected_zip_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0




def load_pkg_index() -> dict[str, str]:
    pkg_index: dict[str, str] = {}
    for pkgdir in (CACHE / "index").iterdir():
        pdir = pkgdir / "packages"
        if not pdir.is_dir():
            continue
        for jf in pdir.glob("*.json"):
            for line in jf.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                pkgid, rest = line.split(" ", 1)
                pkg_index[pkgid] = json.loads(rest)["package"]["archivePath"]
    return pkg_index


def archive_name_from_url(url: str) -> str:
    base = os.path.basename(url)
    if base.endswith(".zip"):
        base = base[:-4]
    return base.replace("%2B", "+")


def valid_ext_dir(path: Path) -> bool:
    return (path / "config" / "extension.toml").is_file() or (path / "extension.toml").is_file()


def flatten_unpack(dest: Path) -> None:
    kids = [p for p in dest.iterdir() if p.name != "PACKAGE-LICENSES"]
    if len(kids) == 1 and kids[0].is_dir() and not valid_ext_dir(dest):
        nested = kids[0]
        for item in nested.iterdir():
            shutil.move(str(item), dest / item.name)
        nested.rmdir()


def add_cache_db_link(cache_path: Path, ext_path: Path) -> None:
    db_file = CACHE / "cache_db.json"
    with db_lock:
        data = {"links_since_last_prune": 0, "links": {}, "last_prune_date": ""}
        if db_file.is_file():
            try:
                data = json.loads(db_file.read_text())
            except json.JSONDecodeError:
                pass
        links = data.setdefault("links", {})
        key = cache_path.name
        norm = os.path.normpath(str(ext_path))
        lst = links.setdefault(key, [])
        if norm not in lst:
            lst.append(norm)
        data["links_since_last_prune"] = int(data.get("links_since_last_prune", 0)) + 1
        db_file.write_text(json.dumps(data, indent=4))


def download_zip(url: str, zpath: Path, expected_size: int) -> str | None:
    """Download extension zip with integrity check. Returns error or None.

    Deliberately avoids curl/wget resume (-C/-c): on AutoDL the partial
    resume often produces concatenated corrupt zips ("extra bytes" errors).
    Each attempt is a fresh download.
    """
    zpath.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        zpath.unlink(missing_ok=True)
        print(
            f"  download attempt {attempt}/{MAX_DOWNLOAD_RETRIES}: "
            f"expected={expected_size} url={url[-60:]}",
            flush=True,
        )
        for cmd in (
            [
                "wget",
                "-q",
                "--dns-timeout=30",
                "--connect-timeout=30",
                "--read-timeout=7200",
                "--tries=1",
                "-O",
                str(zpath),
                url,
            ],
            [
                "curl",
                "-L",
                "-f",
                "--connect-timeout",
                "30",
                "--max-time",
                "7200",
                "-o",
                str(zpath),
                url,
            ],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                continue
            if not zpath.is_file() or zpath.stat().st_size < 1024:
                continue
            got = zpath.stat().st_size
            if expected_size > 0 and got < expected_size:
                print(
                    f"  retry {attempt}/{MAX_DOWNLOAD_RETRIES}: "
                    f"incomplete {got}/{expected_size} bytes",
                    flush=True,
                )
                continue
            test = subprocess.run(
                ["unzip", "-t", str(zpath)],
                capture_output=True,
                text=True,
            )
            if test.returncode != 0:
                print(
                    f"  retry {attempt}/{MAX_DOWNLOAD_RETRIES}: zip test failed "
                    f"({got} bytes)",
                    flush=True,
                )
                continue
            print(f"  download OK: {got} bytes", flush=True)
            return None
        zpath.unlink(missing_ok=True)
    return f"download failed after {MAX_DOWNLOAD_RETRIES} attempts"


def purge_invalid_cache() -> int:
    """Remove cache dirs that exist but lack extension.toml (empty/corrupt)."""
    removed = 0
    if not CACHE.is_dir():
        return 0
    for d in CACHE.iterdir():
        if not d.is_dir() or d.name == "index":
            continue
        if valid_ext_dir(d):
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
        print(f"[purge] removed invalid cache dir {d.name}", flush=True)
    return removed


def install_one(link_path: Path, url: str) -> tuple[str, str]:
    pkgid = link_path.name
    tgt = Path(os.readlink(link_path)) if link_path.is_symlink() else None
    if tgt is not None and tgt.exists() and valid_ext_dir(tgt) and link_path.exists():
        return "skip", pkgid
    archive_name = archive_name_from_url(url)
    dest = CACHE / archive_name
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zpath = ZIP_DIR / f"{pkgid}.zip"

    if not (dest.is_dir() and valid_ext_dir(dest)):
        exp = expected_zip_size(url)
        need_dl = (not zpath.is_file()) or zpath.stat().st_size < 1024
        if not need_dl and exp > 0 and zpath.stat().st_size < exp:
            need_dl = True
        if need_dl:
            err = download_zip(url, zpath, exp)
            if err:
                return "fail", f"{pkgid}: {err}"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["unzip", "-q", "-o", str(zpath), "-d", str(dest)], capture_output=True, text=True)
        if r.returncode != 0:
            return "fail", f"{pkgid}: unzip {r.stderr[:200]}"
        flatten_unpack(dest)
        if not valid_ext_dir(dest):
            return "fail", f"{pkgid}: invalid after unpack"

    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(dest.resolve(), link_path)
    add_cache_db_link(dest.resolve(), link_path)
    return "ok", pkgid


def main() -> int:
    n_purged = purge_invalid_cache()
    if n_purged:
        print(f"purged_invalid_cache_dirs={n_purged}", flush=True)
    pkg_index = load_pkg_index()
    def link_bad(p: Path) -> bool:
        if not p.is_symlink():
            return False
        tgt = Path(os.readlink(p))
        return (not tgt.exists()) or (not valid_ext_dir(tgt))

    broken = [p for p in EXTS3.iterdir() if link_bad(p)]
    print(f"broken_symlinks={len(broken)} workers={WORKERS}", flush=True)
    tasks = []
    for p in broken:
        url = pkg_index.get(p.name)
        if not url:
            print(f"MISSING_URL {p.name}", flush=True)
            continue
        tasks.append((p, url))

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(install_one, p, url): p.name for p, url in tasks}
        for fut in as_completed(futs):
            status, info = fut.result()
            if status == "ok":
                ok += 1
                print(f"OK {info}", flush=True)
            elif status == "skip":
                print(f"SKIP {info}", flush=True)
            else:
                fail += 1
                print(f"FAIL {info}", flush=True)

    remaining = sum(1 for p in EXTS3.iterdir() if p.is_symlink() and not p.exists())
    print(f"done ok={ok} fail={fail} remaining_broken={remaining}", flush=True)
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
