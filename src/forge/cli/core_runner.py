"""Thin wrapper to invoke the legacy monolith core CLI.

Usage:
  python -m forge.cli.core_runner [args...]

This forwards argv to `forge.steel_model_core`'s `__main__` block, so existing
CLI flags continue to work while we migrate to a dedicated, typed CLI.
"""
from __future__ import annotations

import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    # Forward sys.argv into the monolith's __main__
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    runpy.run_module("forge.steel_model_core", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

