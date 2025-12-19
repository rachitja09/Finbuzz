"""A tiny disk-backed FIFO task queue.

This is intentionally minimal: tasks are queued as JSON lines in a file and a
separate worker (`scripts/worker.py`) reads and executes them.

Not intended as a replacement for Celery in production, but useful for local
background work without starting threads inside Streamlit reruns.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterator

QUEUE_FILE = Path(".queue/tasks.jsonl")
QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)


def enqueue(task: Dict[str, Any]) -> None:
    """Append a JSON task to the queue file."""
    with QUEUE_FILE.open("a", encoding="utf8") as f:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")


def iter_tasks() -> Iterator[Dict[str, Any]]:
    """Yield tasks (and clear the queue)."""
    if not QUEUE_FILE.exists():
        return
    # read all and clear file
    with QUEUE_FILE.open("r", encoding="utf8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue
    # truncate file after reading
    QUEUE_FILE.unlink(missing_ok=True)
