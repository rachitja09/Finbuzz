"""Utility: print_lines

Kept as a tiny helper that can be invoked from the command line. The module
no longer executes on import which makes it safe for test runners and
import-time discovery.
"""
from pathlib import Path
import sys
from typing import Sequence

def print_file_with_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        return 2
    text = p.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines()):
        print(f"{i+1:04d}: {line}")
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    p = argv[0] if argv else "pages/01_Home.py"
    return print_file_with_lines(p)

if __name__ == '__main__':
    raise SystemExit(main())
