"""命令行入口：读取 MPS 文件并记录 SCIP 割平面与 LP 状态。"""

import argparse
import json
import os
import sys

from scip_cut_logger import SCIPCutLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record SCIP cutting planes and LP states for an MPS instance."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input MPS file (e.g. 30n20b8.mps).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Directory to write output CSV files (default: ./output).",
    )
    parser.add_argument(
        "--timelimit",
        "-t",
        type=float,
        default=600.0,
        help="SCIP time limit in seconds (default: 600).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress SCIP standard output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    logger = SCIPCutLogger()
    print(f"Solving {args.input} ...")
    summary = logger.solve(
        mps_path=args.input,
        time_limit=args.timelimit,
        quiet=args.quiet,
        stats_path=os.path.join(args.output, "scip_statistics.txt"),
    )

    output_info = logger.write(args.output, summary=summary)
    summary.update(output_info)

    print("Done.")
    print(f"  Status            : {summary['status']}")
    print(f"  Primal bound      : {summary['primal_bound']}")
    print(f"  Dual bound        : {summary['dual_bound']}")
    print(f"  Gap               : {summary['gap']}")
    print(f"  Nodes             : {summary['n_nodes']}")
    print(f"  Candidate cuts    : {summary['n_candidate_cuts']}")
    print(f"  Applied cuts      : {summary['n_applied_cuts']}")
    print(f"  SCIP applied cuts : {summary['n_applied_cuts_scip']}")
    print(f"  LP states         : {summary['n_lp_states']}")
    print(f"  Sep. transitions  : {summary['n_sep_round_transitions']}")
    print(f"  Output directory  : {args.output}")


if __name__ == "__main__":
    main()
