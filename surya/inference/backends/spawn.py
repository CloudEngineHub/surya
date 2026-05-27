"""Server lifecycle: probe, filelock, sentinel, atexit cleanup.

Pattern: probe `/health` → if alive return handle → else acquire lock, re-probe,
spawn detached, write sentinel, register atexit kill (only the spawner cleans up).
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _cache_dir() -> Path:
    base = Path(os.path.expanduser("~/.cache/datalab/surya"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sentinel_path(backend: str) -> Path:
    return _cache_dir() / f"{backend}_server.json"


def _lock_path(backend: str) -> Path:
    return _cache_dir() / f"{backend}_server.lock"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def probe_health(base_url: str, timeout: float = 1.0) -> bool:
    """Returns True if the server reports healthy at /health."""
    try:
        # llama.cpp returns 200 on /health when ready; vllm returns 200 on /health too.
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base_url}/health")
            return r.status_code == 200
    except Exception:
        return False


def wait_for_health(
    base_url: str, total_timeout: float = 300.0, interval: float = 1.0
) -> bool:
    deadline = time.time() + total_timeout
    while time.time() < deadline:
        if probe_health(base_url):
            return True
        time.sleep(interval)
    return False


def probe_model_id(openai_base: str, timeout: float = 5.0) -> Optional[str]:
    """Returns the model id reported by the running server, or None on failure."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{openai_base}/models")
            r.raise_for_status()
            data = r.json()
            models = data.get("data") or []
            if models:
                return models[0].get("id")
    except Exception:
        return None
    return None


@dataclass
class SpawnedServer:
    base_url: str  # full openai base, e.g. "http://127.0.0.1:8765/v1"
    health_url: str  # base for /health, e.g. "http://127.0.0.1:8765"
    model_name: str  # what to pass as `model`
    pid: Optional[int]
    backend: str
    spawned_by_us: bool


class SpawnError(RuntimeError):
    pass


def _read_sentinel(backend: str) -> Optional[dict]:
    p = _sentinel_path(backend)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_sentinel(backend: str, data: dict) -> None:
    _sentinel_path(backend).write_text(json.dumps(data))


def _delete_sentinel(backend: str) -> None:
    p = _sentinel_path(backend)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def _stop_process(pid: int, name: str) -> None:
    try:
        # Graceful first
        os.kill(pid, 15)  # SIGTERM
        for _ in range(20):
            try:
                os.kill(pid, 0)  # still alive?
            except ProcessLookupError:
                logger.info(f"Stopped {name} (pid {pid})")
                return
            time.sleep(0.5)
        # Hard
        os.kill(pid, 9)
        logger.warning(f"Force-killed {name} (pid {pid})")
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Failed to stop {name} (pid {pid}): {e}")


def _stop_docker_container(name: str) -> None:
    try:
        subprocess.run(
            ["docker", "stop", name], check=False, capture_output=True, timeout=30
        )
        logger.info(f"Stopped docker container {name}")
    except Exception as e:
        logger.warning(f"Failed to stop docker container {name}: {e}")


def attach_or_spawn(
    backend: str,
    expected_model_name: str,
    spawn_fn: Callable[[int], "SpawnHandle"],
    health_url_for: Callable[[int], str],
    openai_url_for: Callable[[int], str],
    startup_timeout: float = 600.0,
) -> SpawnedServer:
    """Generic attach-or-spawn with file lock and sentinel.

    `spawn_fn(port)` must launch the server detached and return a SpawnHandle
    with `pid` (int or None for docker) and a `cleanup_id` (e.g. container name).
    """
    # 0. If user pinned an external URL, attach without lock
    if settings.SURYA_INFERENCE_URL:
        base_url = settings.SURYA_INFERENCE_URL.rstrip("/")
        health_url = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
        if not probe_health(health_url):
            raise SpawnError(
                f"SURYA_INFERENCE_URL={base_url} is not reachable at /health. "
                "Start the server or unset the variable."
            )
        model_name = probe_model_id(base_url) or expected_model_name
        if model_name != expected_model_name:
            raise SpawnError(
                f"Model mismatch at {base_url}: expected {expected_model_name!r}, got {model_name!r}. "
                "Stop the running server or unset SURYA_INFERENCE_URL."
            )
        return SpawnedServer(
            base_url=base_url,
            health_url=health_url,
            model_name=model_name,
            pid=None,
            backend=backend,
            spawned_by_us=False,
        )

    # 1. Probe sentinel without lock
    existing = _read_sentinel(backend)
    if existing:
        port = existing.get("port")
        pid = existing.get("pid")
        if port and probe_health(health_url_for(port)):
            running_model = probe_model_id(openai_url_for(port)) or expected_model_name
            if running_model != expected_model_name:
                raise SpawnError(
                    f"Existing {backend} server on port {port} serves {running_model!r}, "
                    f"expected {expected_model_name!r}. Stop it before continuing."
                )
            logger.info(f"Attaching to existing {backend} server on port {port}")
            return SpawnedServer(
                base_url=openai_url_for(port),
                health_url=health_url_for(port),
                model_name=running_model,
                pid=pid,
                backend=backend,
                spawned_by_us=False,
            )
        else:
            _delete_sentinel(backend)

    if not settings.SURYA_INFERENCE_AUTOSTART:
        raise SpawnError(
            f"No running {backend} server and SURYA_INFERENCE_AUTOSTART is False. "
            "Set the variable to True or start the server manually."
        )

    # 2. Acquire filelock to prevent races
    try:
        from filelock import FileLock
    except ImportError as e:
        raise SpawnError(
            "filelock is required for server spawn. pip install filelock"
        ) from e

    lock = FileLock(str(_lock_path(backend)), timeout=120)
    with lock:
        # Re-check sentinel inside the lock
        existing = _read_sentinel(backend)
        if existing:
            port = existing.get("port")
            if port and probe_health(health_url_for(port)):
                running_model = (
                    probe_model_id(openai_url_for(port)) or expected_model_name
                )
                if running_model != expected_model_name:
                    raise SpawnError(
                        f"Existing {backend} server on port {port} serves {running_model!r}, "
                        f"expected {expected_model_name!r}."
                    )
                return SpawnedServer(
                    base_url=openai_url_for(port),
                    health_url=health_url_for(port),
                    model_name=running_model,
                    pid=existing.get("pid"),
                    backend=backend,
                    spawned_by_us=False,
                )

        # 3. Spawn fresh
        port = settings.SURYA_INFERENCE_PORT or find_free_port()
        logger.info(f"Spawning {backend} server on port {port}")
        spawn_handle = spawn_fn(port)

        # 4. Write sentinel
        _write_sentinel(
            backend,
            {
                "port": port,
                "pid": spawn_handle.pid,
                "model": expected_model_name,
                "backend": backend,
                "cleanup_id": spawn_handle.cleanup_id,
                "cleanup_kind": spawn_handle.cleanup_kind,
            },
        )

        # 5. Register atexit cleanup (only spawner). Skipped when keep-alive is
        # set so the server outlives this process and later commands attach to
        # it via the sentinel. (_cleanup is still callable below on startup
        # failure, where we always tear a half-started server down.)
        def _cleanup():
            try:
                if spawn_handle.cleanup_kind == "docker":
                    _stop_docker_container(spawn_handle.cleanup_id)
                elif spawn_handle.cleanup_kind == "process":
                    if spawn_handle.pid:
                        _stop_process(spawn_handle.pid, backend)
            finally:
                _delete_sentinel(backend)

        if settings.SURYA_INFERENCE_KEEP_ALIVE:
            logger.info(
                f"keep-alive: {backend} server on port {port} will stay up "
                f"after exit (cleanup_id={spawn_handle.cleanup_id!r})"
            )
        else:
            atexit.register(_cleanup)

        # 6. Wait for health
        health_url = health_url_for(port)
        if not wait_for_health(health_url, total_timeout=startup_timeout):
            _cleanup()
            raise SpawnError(
                f"{backend} server failed to become healthy at {health_url} "
                f"within {startup_timeout}s"
            )

        # 7. Verify model name
        running_model = probe_model_id(openai_url_for(port))
        if running_model and running_model != expected_model_name:
            logger.warning(
                f"{backend} server reports model={running_model!r} "
                f"but expected {expected_model_name!r}; using reported name."
            )
            expected_model_name = running_model

        logger.info(
            f"{backend} server ready on port {port} (model={expected_model_name})"
        )
        return SpawnedServer(
            base_url=openai_url_for(port),
            health_url=health_url,
            model_name=expected_model_name,
            pid=spawn_handle.pid,
            backend=backend,
            spawned_by_us=True,
        )


@dataclass
class SpawnHandle:
    pid: Optional[int]
    cleanup_id: str  # container name for docker, str(pid) for process
    cleanup_kind: str  # "docker" | "process"
