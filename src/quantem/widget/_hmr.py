"""
HMR (hot module reload) auto-start helper.

Spawns `npm run dev` as a background subprocess so edits to js/*.tsx hot-reload
into live notebooks without a manual "open a second terminal" step.  Also sets
ANYWIDGET_HMR=1 in the current process so anywidget picks up file watcher events.

Usage from a notebook:

    from quantem.widget import enable_hmr
    enable_hmr()

Safe to call multiple times: detects an already-running `npm run dev` process
and skips spawning a second one.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Optional


_PID_FILE = pathlib.Path.home() / ".cache" / "quantem-widget" / "hmr.pid"
_LOG_FILE = pathlib.Path.home() / ".cache" / "quantem-widget" / "hmr.log"
_proc: Optional[subprocess.Popen] = None


def _repo_root() -> pathlib.Path:
    # Walk upward from this file until we find a package.json with an esbuild dev script.
    here = pathlib.Path(__file__).resolve()
    for parent in (here, *here.parents):
        pkg = parent / "package.json"
        if pkg.exists() and "esbuild" in pkg.read_text(encoding="utf-8"):
            return parent
    raise RuntimeError(
        "quantem.widget.enable_hmr(): could not locate package.json. "
        "HMR only works in an editable source install of quantem-widget."
    )


def _read_pidfile() -> Optional[int]:
    try:
        return int(_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def enable_hmr(verbose: bool = True) -> Optional[subprocess.Popen]:
    """Start `npm run dev` in the background if not already running.

    Sets ``ANYWIDGET_HMR=1`` in the current Python process so anywidget picks up
    file-watch events from the bundler.  The spawned process survives kernel
    restarts (it writes to ``~/.cache/quantem-widget/hmr.pid``), so the same
    watcher keeps running across sessions.

    Returns the Popen of a freshly-spawned process, or None if one was already
    running.
    """
    global _proc

    os.environ["ANYWIDGET_HMR"] = "1"

    existing_pid = _read_pidfile()
    if existing_pid and _is_alive(existing_pid):
        if verbose:
            print(f"quantem.widget HMR: already running (pid {existing_pid}).")
            print(f"  log: {_LOG_FILE}")
        return None

    npm = shutil.which("npm")
    if npm is None:
        if verbose:
            print("quantem.widget HMR: `npm` not on PATH; skipping.", file=sys.stderr)
        return None

    try:
        root = _repo_root()
    except RuntimeError as exc:
        if verbose:
            print(f"quantem.widget HMR: {exc}", file=sys.stderr)
        return None

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Open log fresh each spawn so the watcher's output doesn't accumulate forever.
    log_fh = open(_LOG_FILE, "w", buffering=1, encoding="utf-8")
    log_fh.write(f"# quantem.widget HMR started at {time.ctime()}\n")
    log_fh.flush()

    # start_new_session so the child isn't killed with the kernel;
    # this matches the "survive kernel restart" intent.
    _proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(root),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _PID_FILE.write_text(str(_proc.pid))

    if verbose:
        print(f"quantem.widget HMR: started `npm run dev` (pid {_proc.pid}).")
        print(f"  root: {root}")
        print(f"  log:  {_LOG_FILE}")
        print("  Edits to js/*.tsx will rebuild and hot-reload live widgets.")
    return _proc


def disable_hmr(verbose: bool = True) -> None:
    """Stop the background `npm run dev` process started by enable_hmr()."""
    pid = _read_pidfile()
    if pid and _is_alive(pid):
        try:
            os.kill(pid, 15)  # SIGTERM
            if verbose:
                print(f"quantem.widget HMR: stopped pid {pid}.")
        except OSError as exc:
            if verbose:
                print(f"quantem.widget HMR: could not stop pid {pid}: {exc}", file=sys.stderr)
    try:
        _PID_FILE.unlink()
    except OSError:
        pass


def _atexit_note() -> None:
    # Leave the watcher running across kernel restarts; only stop it if the caller
    # explicitly invoked disable_hmr().  Just flush any pending log output.
    pass


atexit.register(_atexit_note)
