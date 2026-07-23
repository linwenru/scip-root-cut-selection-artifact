"""批量跑 benchmark-v2.test 中列出的实例。"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_instance(instance_path, output_dir, timelimit, timeout_buffer, quiet):
    """对单个实例调用 main.py。"""
    name = Path(instance_path).stem
    print(f"Starting {name} ...", flush=True)
    if name.endswith(".mps"):
        name = name[:-4]
    out = os.path.join(output_dir, name)
    os.makedirs(out, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        "-i",
        instance_path,
        "-o",
        out,
        "-t",
        str(timelimit),
    ]
    if quiet:
        cmd.append("--quiet")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timelimit + timeout_buffer,
        )
        elapsed = time.time() - start
        return {
            "name": name,
            "status": "ok" if result.returncode == 0 else "error",
            "elapsed": elapsed,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "timeout",
            "elapsed": timelimit + timeout_buffer,
            "returncode": -1,
            "stdout": "",
            "stderr": "subprocess.TimeoutExpired",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run all instances listed in a benchmark file."
    )
    parser.add_argument(
        "--instances-dir",
        "-d",
        default="instances",
        help="Directory containing .mps.gz instance files.",
    )
    parser.add_argument(
        "--benchmark-list",
        "-b",
        default="benchmark-v2.test",
        help="File listing instance names, one per line.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Use split/<split>.test as the benchmark list (overrides --benchmark-list).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="benchmark_output",
        help="Root directory for per-instance outputs.",
    )
    parser.add_argument(
        "--timelimit",
        "-t",
        type=float,
        default=600.0,
        help="SCIP time limit per instance in seconds.",
    )
    parser.add_argument(
        "--timeout-buffer",
        type=float,
        default=120.0,
        help="Extra wall-clock time (seconds) after SCIP limit for I/O and teardown.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress SCIP output from each instance.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances that already have output_dir/<name>/candidate_cuts.csv.",
    )
    args = parser.parse_args()

    benchmark_list = args.benchmark_list
    if args.split:
        benchmark_list = os.path.join("split", f"{args.split}.test")

    with open(benchmark_list) as f:
        names = [line.strip() for line in f if line.strip()]

    tasks = []
    for name in names:
        path = os.path.join(args.instances_dir, name)
        if not os.path.exists(path):
            print(f"Warning: instance not found, skipping: {path}")
            continue
        tasks.append(path)

    if not tasks:
        print("No instances to run.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary = {}

    # resume 时加载已有 summary
    if args.resume and os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                summary[row["name"]] = row

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_instance, p, args.output_dir, args.timelimit, args.timeout_buffer, args.quiet): p
                for p in tasks
            }
            for future in as_completed(futures):
                res = future.result()
                summary[res["name"]] = res
                print(f"{res['name']}: {res['status']} ({res['elapsed']:.1f}s)")
    else:
        for p in tasks:
            name = Path(p).stem
            if name.endswith(".mps"):
                name = name[:-4]
            if args.resume and os.path.exists(
                os.path.join(args.output_dir, name, "summary.json")
            ):
                if name not in summary:
                    summary[name] = {
                        "name": name,
                        "status": "ok",
                        "elapsed": "",
                        "returncode": "0",
                        "stderr": "",
                    }
                print(f"Skipping {name} (already exists)")
                continue
            res = run_instance(p, args.output_dir, args.timelimit, args.timeout_buffer, args.quiet)
            summary[res["name"]] = res
            print(f"{res['name']}: {res['status']} ({res['elapsed']:.1f}s)")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "status", "elapsed", "returncode", "stderr"]
        )
        writer.writeheader()
        for res in summary.values():
            writer.writerow({
                "name": res["name"],
                "status": res["status"],
                "elapsed": res["elapsed"],
                "returncode": res["returncode"],
                "stderr": res["stderr"][:500] if res["stderr"] else "",
            })

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
