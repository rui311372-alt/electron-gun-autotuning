"""Run the unchanged offline program, writing each run into a new folder."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=base / "candidate_data.csv")
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"Input file not found: {source}")
    output = base / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-X", "utf8", str(base / "stable_working_point_round10.py"), str(source)]
    for option, filename in (
        ("--output-json", "results.json"),
        ("--output-csv", "optimal_points.csv"),
        ("--segments-csv", "segments.csv"),
        ("--sensitivity-csv", "sensitivity.csv"),
        ("--prefix-csv", "prefix.csv"),
    ):
        command.extend([option, str(output / filename)])
    print(f"Input: {source}\nOutput: {output}", flush=True)
    print("Running main analysis and the original sensitivity analyses...", flush=True)
    result = subprocess.run(command, cwd=base, check=False)
    if result.returncode:
        print("The program did not complete. Check the error above; outputs may be partial.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
