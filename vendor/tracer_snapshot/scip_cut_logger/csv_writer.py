"""将记录输出为 CSV 与 JSON 摘要。"""

import json
import os

import pandas as pd


CANDIDATE_COLUMNS = [
    "run_number",
    "node_number",
    "node_depth",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "root",
    "cut_id",
    "cut_name",
    "origin_type",
    "is_forced",
    "is_selected",
    "is_applied",
    "original_index",
    "rank",
    "score",
    "nnz",
    "rhs",
    "lhs",
    "constant",
    "coeff_norm_l2",
    "coeff_norm_l1",
    "coeff_max_abs",
    "coeff_min_abs",
    "coeff_mean_abs",
    "coeff_std_abs",
    "coeff_sparsity_ratio",
    "efficacy",
    "obj_parallelism",
    "cutoff_distance",
    "n_int_cols",
    "is_local",
    "is_modifiable",
    "is_removable",
    "is_integral",
    "in_global_cutpool",
    "lp_position",
]

APPLIED_COLUMNS = [
    "run_number",
    "node_number",
    "node_depth",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "cut_id",
    "cut_name",
    "origin_type",
    "nnz",
    "rhs",
    "lhs",
    "constant",
    "coeff_norm_l2",
    "coeff_norm_l1",
    "coeff_max_abs",
    "coeff_min_abs",
    "coeff_mean_abs",
    "coeff_std_abs",
    "coeff_sparsity_ratio",
    "efficacy",
    "obj_parallelism",
    "cutoff_distance",
    "n_int_cols",
    "is_local",
    "is_modifiable",
    "is_removable",
    "is_integral",
    "in_global_cutpool",
    "lp_position",
]

LP_COLUMNS = [
    "run_number",
    "node_number",
    "node_depth",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "n_lp_rows",
    "n_lp_cols",
    "lp_status",
    "lp_obj_val",
    "lp_iterations",
    "lp_iterations_total",
    "lp_iterations_node",
    "lp_tierations_strongbranch",
    "dual_bound",
    "primal_bound",
    "gap",
    "n_processed_nodes_global",
    "n_processed_nodes_run",
    "n_open_nodes",
    "n_cuts_generated_global",
    "n_cuts_generated_run",
    "n_cuts_generated_node",
    "n_cuts_applied",
    "solving_time",
]

SEP_ROUND_TRANSITION_COLUMNS = [
    "run_number",
    "node_number",
    "node_depth",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "pre_lp_round_global",
    "post_lp_round_global",
    "pre_lp_status",
    "post_lp_status",
    "n_lp_rows_pre",
    "n_lp_rows_post",
    "delta_n_lp_rows",
    "n_lp_cols_pre",
    "n_lp_cols_post",
    "delta_n_lp_cols",
    "lp_obj_val_pre",
    "lp_obj_val_post",
    "delta_lp_obj_val",
    "lp_obj_improvement_ratio",
    "lp_iterations_total_pre",
    "lp_iterations_total_post",
    "delta_lp_iterations_total",
    "lp_iterations_node_pre",
    "lp_iterations_node_post",
    "delta_lp_iterations_node",
    "dual_bound_pre",
    "dual_bound_post",
    "delta_dual_bound",
    "primal_bound_pre",
    "primal_bound_post",
    "delta_primal_bound",
    "gap_pre",
    "gap_post",
    "delta_gap",
    "relative_gap_improvement",
    "n_cuts_applied_pre",
    "n_cuts_applied_post",
    "delta_n_cuts_applied",
    "n_candidate_cuts",
    "n_applied_cuts_round",
    "n_cuts_generated_node_pre",
    "n_cuts_generated_node_post",
    "delta_n_cuts_generated_node",
    "n_open_nodes_pre",
    "n_open_nodes_post",
    "delta_n_open_nodes",
    "n_lp_solves_in_round",
]


def _normalize(records, columns):
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def write_outputs(output_dir, candidate_records, applied_records, lp_records, sep_round_transitions=None, summary=None):
    """将记录写入 output_dir 目录下的 CSV 文件。"""
    os.makedirs(output_dir, exist_ok=True)

    candidate_path = os.path.join(output_dir, "candidate_cuts.csv")
    applied_path = os.path.join(output_dir, "applied_cuts.csv")
    lp_path = os.path.join(output_dir, "lp_states.csv")
    transitions_path = os.path.join(output_dir, "sep_round_transitions.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    df_candidate = _normalize(candidate_records, CANDIDATE_COLUMNS)
    df_applied = _normalize(applied_records, APPLIED_COLUMNS)
    df_lp = _normalize(lp_records, LP_COLUMNS)
    df_transitions = _normalize(sep_round_transitions if sep_round_transitions is not None else [], SEP_ROUND_TRANSITION_COLUMNS)

    df_candidate.to_csv(candidate_path, index=False)
    df_applied.to_csv(applied_path, index=False)
    df_lp.to_csv(lp_path, index=False)
    df_transitions.to_csv(transitions_path, index=False)

    if summary is not None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "candidate_cuts_csv": candidate_path,
        "applied_cuts_csv": applied_path,
        "lp_states_csv": lp_path,
        "sep_round_transitions_csv": transitions_path,
        "summary_json": summary_path if summary is not None else None,
        "n_candidate_cuts": len(df_candidate),
        "n_applied_cuts": len(df_applied),
        "n_lp_states": len(df_lp),
        "n_sep_round_transitions": len(df_transitions),
    }
