#!/usr/bin/env python3
"""
Run named profiles from configs/run_profiles.yml with friendly logging.

Usage
  python3 scripts/run_profiles.py finished
  python3 scripts/run_profiles.py finished paper --parallel
  python3 scripts/run_profiles.py --list

Profiles are editable in configs/run_profiles.yml (env + cmd per profile).
Logs are written under results/<label>/run.log if FORGE_OUTPUT_LABEL is set in env;
otherwise in results/<profile>/run.log.
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess as sp
from pathlib import Path
import yaml
from datetime import datetime


def load_profiles(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profs = data.get("profiles") or {}
    if not isinstance(profs, dict):
        raise ValueError("profiles must be a mapping in run_profiles.yml")
    return profs


def ensure_log_path(env: dict, profile: str) -> Path:
    label = env.get("FORGE_OUTPUT_LABEL") or profile
    log_dir = Path("results") / label
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "run.log"


def run_profile(name: str, spec: dict, *, parallel: bool = False) -> int:
    env = os.environ.copy()
    env_spec = spec.get("env") or {}
    if not isinstance(env_spec, dict):
        raise ValueError(f"profile {name}: env must be a mapping")
    env.update({str(k): str(v) for k, v in env_spec.items()})
    cmd = spec.get("cmd")
    if not isinstance(cmd, list) or not cmd:
        raise ValueError(f"profile {name}: cmd must be a non-empty list")

    log_path = ensure_log_path(env, name)
    print(f"→ Running profile '{name}' → {' '.join(cmd)}")
    print(f"  log: {log_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", buffering=1, encoding="utf-8")
    log_fh.write(f"\n=== START {name} at {datetime.now().isoformat()} ===\n")

    if parallel:
        proc = sp.Popen(cmd, stdout=log_fh, stderr=sp.STDOUT, env=env)
        return proc.pid  # return PID marker; caller will not wait
    else:
        try:
            res = sp.run(cmd, stdout=log_fh, stderr=sp.STDOUT, env=env, check=False)
            code = int(res.returncode or 0)
            return code
        finally:
            log_fh.write(f"\n=== END {name} at {datetime.now().isoformat()} ===\n")
            log_fh.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run named profiles with configured env + commands")
    p.add_argument("profiles", nargs="*", help="Profile names to run (from configs/run_profiles.yml)")
    p.add_argument("--list", action="store_true", help="List available profiles and exit")
    p.add_argument("--parallel", action="store_true", help="Run multiple profiles in parallel")
    args = p.parse_args(argv)

    yml = Path("configs/run_profiles.yml")
    if not yml.exists():
        print("configs/run_profiles.yml not found", file=sys.stderr)
        return 2
    profs = load_profiles(yml)

    if args.list or not args.profiles:
        print("Available profiles:")
        for k, v in profs.items():
            desc = v.get("desc") or ""
            print(f"  - {k}: {desc}")
        return 0 if args.list else 1

    codes: list[int] = []
    if args.parallel and len(args.profiles) > 1:
        pids = []
        for name in args.profiles:
            spec = profs.get(name)
            if not spec:
                print(f"Unknown profile: {name}", file=sys.stderr)
                codes.append(2); continue
            pid = run_profile(name, spec, parallel=True)
            pids.append((name, pid))
        print("Launched:")
        for name, pid in pids:
            print(f"  {name} (pid {pid})")
        print("Waiting… (check logs under results/<label>/run.log)")
        # Not waiting for processes: OS will manage; return success
        return 0
    else:
        for name in args.profiles:
            spec = profs.get(name)
            if not spec:
                print(f"Unknown profile: {name}", file=sys.stderr)
                codes.append(2); continue
            code = run_profile(name, spec, parallel=False)
            codes.append(code)
        return 0 if all(c == 0 for c in codes) else 5


if __name__ == "__main__":
    raise SystemExit(main())

