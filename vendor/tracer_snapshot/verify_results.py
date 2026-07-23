"""核对 CSV 输出与 SCIP printStatistics 的统计信息。"""

import json
import os
import re
import sys

import pandas as pd


def parse_scip_statistics(path):
    """从 SCIP printStatistics 文本中提取关键指标。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    stats = {}

    stats["optimal"] = "problem is solved [optimal solution found]" in text

    m = re.search(r"Total Time\s+:\s+([\d.]+)", text)
    stats["total_time"] = float(m.group(1)) if m else None

    m = re.search(r"solving\s+:\s+([\d.]+)", text)
    stats["solving_time"] = float(m.group(1)) if m else None

    m = re.search(r"nodes \(total\)\s+:\s+(\d+)", text)
    stats["nodes_total"] = int(m.group(1)) if m else None

    m = re.search(r"max depth \(total\)\s*:\s*(\d+)", text)
    stats["max_depth_total"] = int(m.group(1)) if m else None

    m = re.search(r"Primal Bound\s+:\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", text)
    stats["primal_bound"] = float(m.group(1)) if m else None

    m = re.search(r"^\s*Dual Bound\s+:\s+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", text, re.MULTILINE)
    stats["dual_bound"] = float(m.group(1)) if m else None

    m = re.search(r"Gap\s+:\s+([\d.]+)\s+%", text)
    stats["gap_percent"] = float(m.group(1)) if m else None

    # LP 迭代（LP 小节）
    lp_section = re.search(r"^LP\s+:(.*?)(?:^Relaxators|:B&B Tree)", text, re.MULTILINE | re.DOTALL)
    if lp_section:
        lp_text = lp_section.group(1)
        lp_patterns = [
            (r"dual LP", "lp_dual"),
            (r"diving/probing LP", "lp_diving"),
            (r"strong branching", "lp_strongbr"),
            (r"conflict analysis", "lp_conflict"),
        ]
        for label, key in lp_patterns:
            m = re.search(
                rf"^\s+{label}\s*:\s+[\d.]+\s+(\d+)\s+(\d+)",
                lp_text,
                re.MULTILINE,
            )
            if m:
                stats[f"{key}_calls"] = int(m.group(1))
                stats[f"{key}_iterations"] = int(m.group(2))
            else:
                stats[f"{key}_calls"] = None
                stats[f"{key}_iterations"] = None
    else:
        for key in ["lp_dual", "lp_diving", "lp_strongbr", "lp_conflict"]:
            stats[f"{key}_calls"] = None
            stats[f"{key}_iterations"] = None

    stats["lp_total_iterations"] = sum(
        v for v in [
            stats.get("lp_dual_iterations"),
            stats.get("lp_diving_iterations"),
            stats.get("lp_strongbr_iterations"),
            stats.get("lp_conflict_iterations"),
        ] if v is not None
    )

    # Cutselector 统计：解析自定义 selector 与默认 hybrid selector
    m = re.search(
        r"py_cutsel_logger\s+:\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        text,
    )
    if m:
        stats["cutsel_calls"] = int(m.group(1))
        stats["cutsel_root_calls"] = int(m.group(2))
        stats["cutsel_selected"] = int(m.group(3))
        stats["cutsel_forced"] = int(m.group(4))
        stats["cutsel_filtered"] = int(m.group(5))
    else:
        stats["cutsel_calls"] = None
        stats["cutsel_root_calls"] = None
        stats["cutsel_selected"] = None
        stats["cutsel_forced"] = None
        stats["cutsel_filtered"] = None

    m = re.search(
        r"hybrid\s+:\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        text,
    )
    if m:
        stats["hybrid_calls"] = int(m.group(1))
        stats["hybrid_root_calls"] = int(m.group(2))
        stats["hybrid_selected"] = int(m.group(3))
        stats["hybrid_forced"] = int(m.group(4))
        stats["hybrid_filtered"] = int(m.group(5))
    else:
        stats["hybrid_calls"] = None
        stats["hybrid_root_calls"] = None
        stats["hybrid_selected"] = None
        stats["hybrid_forced"] = None
        stats["hybrid_filtered"] = None

    # Separators 统计（仅解析 Separators 小节）
    sep_section = re.search(r"^Separators\s+:(.*?)(?:^Cutselectors)", text, re.MULTILINE | re.DOTALL)
    separators = []
    if sep_section:
        sep_text = sep_section.group(1)
        sep_pattern = re.compile(
            r"^\s+(\S+)\s+:\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
            re.MULTILINE,
        )
        for m in sep_pattern.finditer(sep_text):
            name = m.group(1)
            if name.startswith(">") or name == "cut":
                continue
            separators.append({
                "name": name,
                "calls": int(m.group(3)),
                "root_calls": int(m.group(4)),
                "found": int(m.group(7)),
                "via_pool_add": int(m.group(8)),
                "direct_add": int(m.group(9)),
                "applied": int(m.group(10)),
                "via_pool_app": int(m.group(11)),
                "direct_app": int(m.group(12)),
            })
    stats["separators"] = separators

    return stats


def check_n_cuts_applied_consistency(app, lp):
    """验证 lp_states.n_cuts_applied 与 applied_cuts.csv 累计行数是否一致。

    子节点 LP 会继承父节点的割，因此 n_cuts_applied = inherited_cuts[node]
    + cumulative_applied_at_node_up_to_sep_round。
    其中 inherited_cuts[node] 由节点首个 LP 状态的 n_cuts_applied 反推得到。
    """
    if len(lp) == 0 or len(app) == 0:
        return {
            "match_rate": None,
            "matches": 0,
            "total": 0,
            "mismatches": pd.DataFrame(),
        }

    app_counts = (
        app.groupby(["run_number", "node_number", "sep_round_node"])
        .size()
        .reset_index(name="n_applied")
        .sort_values(["run_number", "node_number", "sep_round_node"])
    )
    app_counts["cum_applied"] = app_counts.groupby(["run_number", "node_number"])["n_applied"].cumsum()

    lp_sorted = lp.sort_values(["run_number", "node_number", "lp_round_node"])
    first_idx = lp_sorted.groupby(["run_number", "node_number"])["lp_round_node"].idxmin()
    first_lp = lp_sorted.loc[first_idx].copy()
    first_lp = first_lp[["run_number", "node_number", "sep_round_node", "n_cuts_applied"]].rename(
        columns={"sep_round_node": "first_sep", "n_cuts_applied": "base_n_cuts"}
    )

    first_cum = app_counts.merge(
        first_lp.rename(columns={"first_sep": "sep_round_node"}),
        on=["run_number", "node_number", "sep_round_node"],
        how="inner",
    )[["run_number", "node_number", "cum_applied"]].rename(columns={"cum_applied": "first_cum"})
    first_lp = first_lp.merge(first_cum, on=["run_number", "node_number"], how="left")
    first_lp["first_cum"] = first_lp["first_cum"].fillna(0)
    first_lp["inherited"] = first_lp["base_n_cuts"] - first_lp["first_cum"]

    lp_merged = lp.merge(
        app_counts[["run_number", "node_number", "sep_round_node", "cum_applied"]],
        on=["run_number", "node_number", "sep_round_node"],
        how="left",
    )
    lp_merged["cum_applied"] = lp_merged["cum_applied"].fillna(0)
    lp_merged = lp_merged.merge(
        first_lp[["run_number", "node_number", "inherited"]],
        on=["run_number", "node_number"],
        how="left",
    )
    lp_merged["inherited"] = lp_merged["inherited"].fillna(0)
    lp_merged["expected_n_cuts_applied"] = lp_merged["inherited"] + lp_merged["cum_applied"]
    lp_merged["match"] = lp_merged["n_cuts_applied"] == lp_merged["expected_n_cuts_applied"]

    matches = int(lp_merged["match"].sum())
    total = len(lp_merged)
    return {
        "match_rate": matches / total if total > 0 else None,
        "matches": matches,
        "total": total,
        "mismatches": lp_merged[~lp_merged["match"]][[
            "run_number", "node_number", "sep_round_node", "lp_round_node",
            "n_cuts_applied", "expected_n_cuts_applied"
        ]],
    }


def summarize_csv(output_dir):
    """从 CSV 文件中提取统计信息。"""
    summary = {}

    cand = pd.read_csv(
        os.path.join(output_dir, "candidate_cuts.csv"),
        dtype={"is_selected": "boolean"},
    )
    app = pd.read_csv(os.path.join(output_dir, "applied_cuts.csv"))
    lp = pd.read_csv(os.path.join(output_dir, "lp_states.csv"))

    summary["candidate_rows"] = len(cand)
    summary["candidate_sepa"] = int((cand["origin_type"] == "SEPA").sum())
    summary["candidate_conshdlr"] = int((cand["origin_type"] == "CONSHDLR").sum())
    summary["candidate_cons"] = int((cand["origin_type"] == "CONS").sum())
    summary["candidate_other_type"] = int((~cand["origin_type"].isin(["SEPA", "CONSHDLR", "CONS"])).sum())
    summary["candidate_selected"] = int(cand["is_selected"].sum())
    summary["candidate_forced"] = int(cand["is_forced"].sum())
    summary["candidate_applied"] = int(cand["is_applied"].sum())
    summary["candidate_unique_ids"] = cand["cut_id"].nunique()
    summary["candidate_nodes"] = cand["node_number"].nunique()
    summary["candidate_sep_rounds"] = cand["sep_round_global"].nunique()

    summary["applied_rows"] = len(app)
    summary["applied_unique_ids"] = app["cut_id"].nunique()
    summary["applied_nodes"] = app["node_number"].nunique()

    summary["lp_states_rows"] = len(lp)
    summary["lp_states_nodes"] = lp["node_number"].nunique()
    summary["lp_total_iterations"] = int(lp["lp_iterations_total"].max()) if len(lp) > 0 else 0
    summary["lp_last_obj"] = lp["lp_obj_val"].iloc[-1] if len(lp) > 0 else None
    summary["lp_last_dual_bound"] = lp["dual_bound"].iloc[-1] if len(lp) > 0 else None
    summary["lp_last_primal_bound"] = lp["primal_bound"].iloc[-1] if len(lp) > 0 else None
    summary["lp_last_gap"] = lp["gap"].iloc[-1] if len(lp) > 0 else None

    summary["n_cuts_applied_consistency"] = check_n_cuts_applied_consistency(app, lp)

    return summary


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} OUTPUT_DIR")
        sys.exit(2)
    output_dir = sys.argv[1]

    stats_path = os.path.join(output_dir, "scip_statistics.txt")
    if not os.path.isfile(stats_path):
        print(f"SCIP statistics file not found: {stats_path}")
        sys.exit(1)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        json_summary = json.load(f)

    scip_stats = parse_scip_statistics(stats_path)
    csv_summary = summarize_csv(output_dir)

    print("=" * 70)
    print("CSV / JSON 与 SCIP printStatistics 核对报告")
    print("=" * 70)
    print()

    print("【求解状态】")
    print(f"  SCIP optimal        : {scip_stats['optimal']}")
    print(f"  JSON status         : {json_summary.get('status')}")
    print()

    print("【目标值与界】")
    print(f"  Primal bound (SCIP) : {scip_stats['primal_bound']}")
    print(f"  Primal bound (JSON) : {json_summary.get('primal_bound')}")
    print(f"  Dual bound (SCIP)   : {scip_stats['dual_bound']}")
    print(f"  Dual bound (JSON)   : {json_summary.get('dual_bound')}")
    print(f"  Gap (SCIP)          : {scip_stats['gap_percent']} %")
    print(f"  Gap (JSON)          : {json_summary.get('gap')}")
    print()

    print("【节点与深度】")
    print(f"  Nodes total (SCIP)  : {scip_stats['nodes_total']}")
    print(f"  Nodes (JSON)        : {json_summary.get('n_nodes')}")
    print(f"  LP states 节点数    : {csv_summary['lp_states_nodes']}")
    print(f"  Max depth total     : {scip_stats['max_depth_total']}")
    print()

    print("【时间】")
    print(f"  Total time (SCIP)   : {scip_stats['total_time']}")
    print(f"  Solving time (SCIP) : {scip_stats['solving_time']}")
    print(f"  Solving time (JSON) : {json_summary.get('solving_time')}")
    print()

    print("【LP 迭代】")
    print(f"  dual LP iters       : {scip_stats['lp_dual_iterations']}")
    print(f"  diving/probing iters: {scip_stats['lp_diving_iterations']}")
    print(f"  strong branching    : {scip_stats['lp_strongbr_iterations']}")
    print(f"  conflict analysis   : {scip_stats['lp_conflict_iterations']}")
    print(f"  SCIP LP total iters : {scip_stats['lp_total_iterations']}")
    print(f"  CSV max lp_iterations_total: {csv_summary['lp_total_iterations']}")
    print(f"  JSON n_lp_iterations: {json_summary.get('n_lp_iterations')}")
    print()

    print("【割平面选择器 (Cutselector)】")
    print(f"  py_cutsel_logger Calls      : {scip_stats['cutsel_calls']}")
    print(f"  py_cutsel_logger Root calls : {scip_stats['cutsel_root_calls']}")
    print(f"  py_cutsel_logger Selected   : {scip_stats['cutsel_selected']}")
    print(f"  py_cutsel_logger Forced     : {scip_stats['cutsel_forced']}")
    print(f"  py_cutsel_logger Filtered   : {scip_stats['cutsel_filtered']}")
    if scip_stats.get("hybrid_calls") is not None:
        print(f"  hybrid Calls                : {scip_stats['hybrid_calls']}")
        print(f"  hybrid Root calls           : {scip_stats['hybrid_root_calls']}")
        print(f"  hybrid Selected             : {scip_stats['hybrid_selected']}")
        print(f"  hybrid Forced               : {scip_stats['hybrid_forced']}")
        print(f"  hybrid Filtered             : {scip_stats['hybrid_filtered']}")
        hybrid_total = (
            scip_stats.get("hybrid_selected", 0)
            + scip_stats.get("hybrid_forced", 0)
            + scip_stats.get("hybrid_filtered", 0)
        )
        print(f"  hybrid Selected+Forced+Filtered: {hybrid_total}")
    print()

    print("【已应用割 (application events)】")
    print(f"  CSV candidate rows  : {csv_summary['candidate_rows']}")
    print(f"  CSV SEPA candidates : {csv_summary['candidate_sepa']}")
    print(f"  CSV CONSHDLR cand.  : {csv_summary['candidate_conshdlr']}")
    print(f"  CSV CONS candidates : {csv_summary['candidate_cons']}")
    print(f"  CSV is_selected     : {csv_summary['candidate_selected']}")
    print(f"  CSV is_forced       : {csv_summary['candidate_forced']}")
    print(f"  CSV is_applied      : {csv_summary['candidate_applied']}")
    print(f"  CSV applied rows    : {csv_summary['applied_rows']}")
    print(f"  CSV applied unique  : {csv_summary['applied_unique_ids']}")
    print(f"  JSON SCIP applied   : {json_summary.get('n_applied_cuts_scip')}")
    print()
    cons = csv_summary["n_cuts_applied_consistency"]
    print(f"  LP states 记录数    : {cons['total']}")
    print(f"  n_cuts_applied 一致 : {cons['matches']} / {cons['total']}", end="")
    if cons["match_rate"] is not None:
        print(f"  ({cons['match_rate']*100:.2f}%)")
    else:
        print()
    if cons["mismatches"].shape[0] > 0:
        print("  不一致样例（前 5 条）：")
        print(cons["mismatches"].head(5).to_string(index=False))
    print()

    print("【Separators 详细统计】")
    print(f"  {'Name':<18} {'Calls':>8} {'Root':>8} {'Found':>8} {'Applied':>8}")
    scip_applied_total = 0
    for sep in scip_stats["separators"]:
        print(f"  {sep['name']:<18} {sep['calls']:>8} {sep['root_calls']:>8} {sep['found']:>8} {sep['applied']:>8}")
        scip_applied_total += sep['applied']
    print(f"  {'Total':<18} {'':>8} {'':>8} {'':>8} {scip_applied_total:>8}")
    print()

    print("【一致性判断】")
    checks = []
    checks.append(("Status optimal", scip_stats["optimal"] and json_summary.get("status") == "optimal"))
    checks.append(("Primal bound match", scip_stats["primal_bound"] == json_summary.get("primal_bound")))
    checks.append(("Dual bound match", scip_stats["dual_bound"] == json_summary.get("dual_bound")))
    checks.append(("Gap match", scip_stats["gap_percent"] == json_summary.get("gap")))
    checks.append(("Nodes match", scip_stats["nodes_total"] == json_summary.get("n_nodes")))
    if scip_stats["cutsel_selected"] is not None and scip_stats["cutsel_filtered"] is not None:
        # 被动 selector（返回 DIDNOTFIND）时自定义 selector 不执行选择，
        # 实际选择由默认 hybrid selector 完成；此时 candidate_rows 应接近
        # hybrid 的 Selected + Forced + Filtered（允许少量未选中也未被过滤的割）。
        if scip_stats["cutsel_selected"] == 0 and scip_stats["cutsel_filtered"] == 0 \
                and scip_stats.get("hybrid_selected") is not None:
            hybrid_total = (
                scip_stats["hybrid_selected"]
                + scip_stats.get("hybrid_forced", 0)
                + scip_stats["hybrid_filtered"]
            )
            diff = abs(csv_summary["candidate_rows"] - hybrid_total)
            checks.append(("Candidate cuts ~ hybrid processed",
                           diff <= max(10, 0.01 * csv_summary["candidate_rows"])))
        else:
            checks.append(("Candidate cuts = Selected+Filtered",
                           csv_summary["candidate_rows"] == scip_stats["cutsel_selected"] + scip_stats["cutsel_filtered"]))
    cons = csv_summary["n_cuts_applied_consistency"]
    if cons["match_rate"] is not None:
        checks.append(("n_cuts_applied consistency >= 95%",
                       cons["match_rate"] >= 0.95))
    if csv_summary["applied_unique_ids"] > 0:
        checks.append(("Candidate applied >= applied unique ids",
                       csv_summary["candidate_applied"] >= csv_summary["applied_unique_ids"]))
    checks.append(("Candidate applied <= candidate rows",
                   csv_summary["candidate_applied"] <= csv_summary["candidate_rows"]))
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    print()
    print("注意：")
    print("  - candidate_cuts.csv 中 is_applied 按 (run_number, node_number, sep_round_node, cut_id)")
    print("    匹配；cut_id 已纳入 row.name，可区分同一节点内不同来源但内容相同的行。")
    print("    若 cut_id 缺失，则回退到 (run_number, node_number, sep_round_node, cut_name)。")
    print("  - candidate_cuts.csv 与 applied_cuts.csv 均可能包含 origin_type=CONS 记录，")
    print("    因为 SCIP 会把部分原始约束作为割加入 LP。")
    print("  - applied_cuts.csv 的每一行代表一次割被加入 LP 的事件（行数口径），")
    print("    同一条割在不同节点/轮次被加入会被多次记录；因此它与 SCIP Separators")
    print("    的累计 Applied（去重后的割数量）以及 getNCutsApplied() 的当前 LP 割数")
    print("    口径均不同。")
    print("  - lp_states.csv 中的 n_cuts_applied 表示当前 LP 中已应用的割行数；")
    print("    对根节点它等于 applied_cuts.csv 在该节点的累计行数，对子节点还包含")
    print("    从父节点继承的割。verify 中通过 inherited + cumulative 的方式校验一致性。")
    print("  - CSV 的 lp_iterations_total 取自 SCIP API getNLPIterations，")
    print("    与 printStatistics 中各类 LP 迭代之和口径不同。")


if __name__ == "__main__":
    main()
