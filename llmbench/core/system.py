"""Environment metadata snapshot, attached to every sweep (llama-bench style)."""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess


def _try(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def snapshot() -> dict[str, str]:
    os_name = platform.platform()
    try:
        pretty = platform.freedesktop_os_release().get("PRETTY_NAME", "")
        if pretty:
            os_name = pretty
    except OSError:
        pass
    info: dict[str, str] = {
        "hostname": socket.gethostname(),
        "os": os_name,
        "kernel": platform.release(),
        "cpu": "",
        "cpu_cores": str(os.cpu_count() or ""),
        "mem": "",
        "gpus": "",
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        g = _try(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        info["gpus"] = g.replace("\n", "; ")
    return info
