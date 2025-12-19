"""Simple worker to process tasks enqueued by the app.

Run this in a separate terminal during development:
    python ./scripts/worker.py

It reads tasks from `.queue/tasks.jsonl` and dispatches them to registered handlers.
"""
from __future__ import annotations
import time
from typing import Dict, Any

from utils.task_queue import iter_tasks


def handle_prefetch_profile(task: Dict[str, Any]) -> None:
    # lightweight prefetch example: call data_providers.fmp_profile and ignore result
    try:
        from data_providers import fmp_profile
        sym = task.get("symbol")
        key = task.get("api_key")
        if sym and key is not None:
            print(f"Prefetching profile for {sym}")
            fmp_profile(sym, key)
    except Exception as e:
        print("prefetch failed:", e)


HANDLERS = {
    "prefetch_profile": handle_prefetch_profile,
}


def main():
    print("Worker started. Polling .queue/tasks.jsonl...")
    while True:
        for task in iter_tasks():
            tname = str(task.get("type") or "")
            handler = HANDLERS.get(tname)
            if handler:
                try:
                    handler(task)
                except Exception as e:
                    print(f"Task {tname} failed: {e}")
            else:
                print(f"No handler for task: {tname}")
        time.sleep(2)


if __name__ == "__main__":
    main()
