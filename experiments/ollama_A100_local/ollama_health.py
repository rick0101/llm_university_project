"""Small Ollama health/restart helpers for SLURM runs.

The helpers are intentionally process-level and conservative: check the local
HTTP endpoint first, then terminate only explicit `ollama serve` PIDs before
starting a new daemon.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests


def daemon_alive(host: str, timeout: int = 5) -> bool:
    """Return True when the Ollama daemon answers `/api/tags`."""
    try:
        return requests.get(f"{host}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _serve_pids() -> list[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    return [pid.strip() for pid in out.stdout.splitlines() if pid.strip().isdigit()]


def _terminate_serve_processes() -> None:
    for pid in _serve_pids():
        try:
            subprocess.run(["kill", pid], timeout=5)
            print(f"   Stopped stale ollama PID {pid}", flush=True)
        except Exception:
            pass

    time.sleep(2)

    for pid in _serve_pids():
        try:
            subprocess.run(["kill", "-9", pid], timeout=5)
            print(f"   Killed unresponsive ollama PID {pid}", flush=True)
        except Exception:
            pass


def restart_daemon(host: str, wait_secs: int = 45) -> bool:
    """Restart `ollama serve` and wait until it answers HTTP health checks."""
    print("   Ollama daemon not responding, attempting safe restart...", flush=True)
    _terminate_serve_processes()

    log_dir = Path(os.environ.get("OLLAMA_RESTART_LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"ollama_restart_{os.getpid()}_{ts}.log"
        log_file = open(log_path, "ab")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"   Relaunched ollama serve (log: {log_path})", flush=True)
    except Exception as exc:
        print(f"   Cannot relaunch ollama serve: {exc}", flush=True)
        return False

    for i in range(wait_secs):
        if daemon_alive(host):
            print(f"   Ollama ready again after {i + 1}s", flush=True)
            return True
        time.sleep(1)

    print(f"   Ollama still down after {wait_secs}s", flush=True)
    return False


def ensure_daemon_alive(host: str, timeout: int = 5, wait_secs: int = 45) -> bool:
    """Check Ollama and restart it when the daemon is down."""
    return daemon_alive(host, timeout=timeout) or restart_daemon(host, wait_secs=wait_secs)
