"""SCIP 求解器封装：安装回调并运行求解。"""

import os
from pyscipopt import Model

from .state_tracker import StateTracker
from .callbacks import CutselCallback, EventCallback
from .csv_writer import write_outputs


class SCIPCutLogger:
    """读取 MPS 实例，运行 SCIP，并记录割平面与 LP 状态。"""

    def __init__(self):
        self.state_tracker = StateTracker()
        self.candidate_records = []
        self.applied_records = []
        self.lp_records = []
        self.sep_round_pre_states = []
        self.sep_round_transitions = []

    def solve(self, mps_path, time_limit=3600.0, quiet=False, params=None, stats_path=None):
        """求解给定的 MPS 文件。

        Parameters
        ----------
        mps_path : str
            MPS 文件路径。
        time_limit : float
            求解时间限制（秒）。
        quiet : bool
            是否禁用 SCIP 标准输出。
        params : dict, optional
            额外的 SCIP 参数，例如 {"separating/maxrounds": 5}。
        stats_path : str, optional
            若提供，则在求解后将 SCIP printStatistics() 输出到该文件。

        Returns
        -------
        dict
            求解摘要。
        """
        if not os.path.isfile(mps_path):
            raise FileNotFoundError(f"MPS file not found: {mps_path}")

        model = Model()
        model.readProblem(mps_path)
        model.setParam("limits/time", time_limit)

        if quiet:
            model.hideOutput()

        # 默认保留 SCIP 的割平面流程，但可让用户覆盖参数
        if params:
            for key, value in params.items():
                if isinstance(value, bool):
                    model.setBoolParam(key, value)
                elif isinstance(value, int):
                    model.setIntParam(key, value)
                elif isinstance(value, float):
                    model.setRealParam(key, value)
                else:
                    model.setParam(key, value)

        # 安装割平面选择器（最高优先级，用于捕获候选割）
        cutsel = CutselCallback(self.state_tracker, self.candidate_records, self.sep_round_pre_states)
        model.includeCutsel(
            cutsel,
            "py_cutsel_logger",
            "records all candidate cuts and their rankings",
            99999999,
        )

        # 安装事件处理器（记录 LP 状态与加入 LP 的割）
        event_hdlr = EventCallback(
            self.state_tracker, self.applied_records, self.lp_records
        )
        model.includeEventhdlr(
            event_hdlr,
            "py_event_logger",
            "records LP states and cuts added to the LP",
        )

        model.optimize()

        self._mark_applied_candidates()
        self._backfill_candidate_ranks()
        self._build_sep_round_transitions()

        if stats_path is not None:
            try:
                model.printStatistics(stats_path)
            except Exception as exc:
                # 统计文件输出失败不应中断主流程
                summary["stats_error"] = str(exc)

        status = model.getStatus()
        try:
            n_applied_cuts_scip = int(model.getNCutsApplied())
        except Exception:
            n_applied_cuts_scip = 0
        summary = {
            "status": status,
            "primal_bound": self._safe_float(model.getPrimalbound()),
            "dual_bound": self._safe_float(model.getDualbound()),
            "gap": self._safe_float(model.getGap()),
            "n_nodes": int(model.getNTotalNodes()),
            "n_lp_iterations": int(model.getNLPIterations()),
            "solving_time": self._safe_float(model.getSolvingTime()),
            "n_candidate_cuts": len(self.candidate_records),
            "n_applied_cuts": len(self.applied_records),
            "n_applied_cuts_scip": n_applied_cuts_scip,
            "n_lp_states": len(self.lp_records),
        }

        return summary

    def _mark_applied_candidates(self):
        """求解结束后，根据实际加入 LP 的割标记候选割的 is_applied 字段。

        使用 (run_number, node_number, sep_round_node, cut_id) 作为 key：
        cut_name 能区分同一节点内不同来源但内容相同的行
        （例如 start_24_mcseq0 与 start_24_ewseq0 的 cut_id 冲突问题），
        sep_round_node 保证同一节点多轮分离时按轮次精确匹配。

        对 cut_id 缺失的退行记录，回退到 (run_number, node_number, sep_round_node, cut_name)。
        """
        applied_keys = set()
        for rec in self.applied_records:
            run = rec.get("run_number")
            node = rec.get("node_number")
            sep_round = rec.get("sep_round_node")
            if sep_round == 0:
                continue # 因为candidate_cuts.csv中没有sep_round_node=0的记录，所以这里跳过
            cut_id = rec.get("cut_id")
            if cut_id is not None:
                applied_keys.add((run, node, sep_round, cut_id))
            else:
                applied_keys.add((run, node, sep_round, rec.get("cut_name")))

        for rec in self.candidate_records:
            run = rec.get("run_number")
            node = rec.get("node_number")
            sep_round = rec.get("sep_round_node")
            cut_id = rec.get("cut_id")
            if cut_id is not None:
                key = (run, node, sep_round, cut_id)
            else:
                key = (run, node, sep_round, rec.get("cut_name"))
            rec["is_applied"] = key in applied_keys

    def _backfill_candidate_ranks(self):
        """根据 applied_cuts.csv 的加入顺序回填 candidate_cuts.csv 的 rank。

        对同一个 (run_number, node_number, sep_round_node) 分组：
        - 按 applied_records 中出现的先后顺序为实际加入 LP 的候选割编号；
        - 未被应用的候选割按 score 降序排在后面。
        """
        from collections import defaultdict

        group_key = lambda rec: (
            rec.get("run_number"),
            rec.get("node_number"),
            rec.get("sep_round_node"),
        )

        cand_by_group = defaultdict(list)
        for rec in self.candidate_records:
            cand_by_group[group_key(rec)].append(rec)

        applied_by_group = defaultdict(list)
        for rec in self.applied_records:
            if rec.get("sep_round_node") == 0:
                continue
            applied_by_group[group_key(rec)].append(rec)

        for key, group_cands in cand_by_group.items():
            for rec in group_cands:
                rec["rank"] = None

            applied_list = applied_by_group.get(key, [])

            by_id = {}
            by_name = {}
            for rec in group_cands:
                cid = rec.get("cut_id")
                if cid is not None:
                    by_id.setdefault(cid, rec)
                name = rec.get("cut_name")
                if name is not None:
                    by_name.setdefault(name, rec)

            rank = 0
            for app_rec in applied_list:
                cid = app_rec.get("cut_id")
                if cid is not None and cid in by_id:
                    rec = by_id.pop(cid)
                    by_name.pop(rec.get("cut_name"), None)
                else:
                    name = app_rec.get("cut_name")
                    if name is None or name not in by_name:
                        continue
                    rec = by_name.pop(name)
                    by_id.pop(rec.get("cut_id"), None)

                rank += 1
                rec["rank"] = rank

            unranked = [rec for rec in group_cands if rec["rank"] is None]
            unranked.sort(
                key=lambda r: r.get("score")
                if r.get("score") is not None
                else float("-inf"),
                reverse=True,
            )
            for rec in unranked:
                rank += 1
                rec["rank"] = rank

    def _build_sep_round_transitions(self):
        """根据每轮分离开始时的 LP 快照构造前后状态变化表。

        对每一轮 r（run_number, node_number, sep_round_node）：
        - pre  取该轮开始时的 LP 快照（cutselselect 中记录）；
        - post 优先取同节点下一轮 r+1 的 LP 快照（即第 r 轮结束后的状态），
          否则回退到同节点 sep_round_node = r 或 >r 的第一个 LPSOLVED 状态。
        """
        import pandas as pd
        import numpy as np

        if not self.sep_round_pre_states:
            self.sep_round_transitions = []
            return

        pre_df = pd.DataFrame(self.sep_round_pre_states)
        pre_df = pre_df.sort_values(["run_number", "node_number", "sep_round_node"])

        lp_df = pd.DataFrame(self.lp_records)
        if not lp_df.empty:
            lp_df = lp_df.sort_values(["run_number", "node_number", "lp_round_global"])

        # 候选割、应用割的每轮计数
        cand_counts = pd.Series(dtype=int, name="n_candidate_cuts")
        if self.candidate_records:
            cand_df = pd.DataFrame(self.candidate_records)
            if not cand_df.empty:
                cand_counts = cand_df.groupby(["run_number", "node_number", "sep_round_node"]).size()

        app_counts = pd.Series(dtype=int, name="n_applied_cuts_round")
        if self.applied_records:
            app_df = pd.DataFrame(self.applied_records)
            if not app_df.empty:
                app_counts = app_df.groupby(["run_number", "node_number", "sep_round_node"]).size()

        def numeric_delta(pre_val, post_val):
            pre_f = pd.to_numeric(pre_val, errors="coerce")
            post_f = pd.to_numeric(post_val, errors="coerce")
            if pd.isna(pre_f) or pd.isna(post_f):
                return None
            return post_f - pre_f

        def safe_ratio(diff, base):
            try:
                base_f = float(base)
                if base_f == 0 or not np.isfinite(base_f):
                    return None
                return diff / base_f
            except Exception:
                return None

        def find_post(run, node, r):
            """先找同节点 r+1 的 pre_state，否则找同节点 sep=r 或 >r 的第一个 LP 状态。"""
            next_pre = pre_df[(pre_df.run_number == run) & (pre_df.node_number == node) & (pre_df.sep_round_node > r)]
            if not next_pre.empty:
                return next_pre.sort_values("sep_round_node").iloc[0]

            if lp_df.empty:
                return None

            same = lp_df[(lp_df.run_number == run) & (lp_df.node_number == node) & (lp_df.sep_round_node == r)]
            if not same.empty:
                return same.sort_values("lp_round_global").iloc[0]

            later = lp_df[(lp_df.run_number == run) & (lp_df.node_number == node) & (lp_df.sep_round_node > r)]
            if not later.empty:
                return later.sort_values("lp_round_global").iloc[0]

            return None

        transitions = []
        for _, pre in pre_df.iterrows():
            run = pre["run_number"]
            node = pre["node_number"]
            r = pre["sep_round_node"]

            post = find_post(run, node, r)
            if post is None:
                continue

            n_cands = int(cand_counts.get((run, node, r), 0))
            n_applied_round = int(app_counts.get((run, node, r), 0))
            n_lp_solves = 0
            if not lp_df.empty:
                n_lp_solves = len(lp_df[(lp_df.run_number == run) & (lp_df.node_number == node) & (lp_df.sep_round_node == r)])

            lp_obj_pre = pd.to_numeric(pre["lp_obj_val"], errors="coerce")
            lp_obj_delta = numeric_delta(pre["lp_obj_val"], post["lp_obj_val"])
            lp_obj_ratio = safe_ratio(lp_obj_delta, lp_obj_pre) if lp_obj_delta is not None else None

            gap_pre = pd.to_numeric(pre["gap"], errors="coerce")
            gap_post = pd.to_numeric(post["gap"], errors="coerce")
            gap_delta = numeric_delta(pre["gap"], post["gap"])
            relative_gap_improvement = None
            if gap_delta is not None and gap_pre is not None and pd.notna(gap_pre):
                relative_gap_improvement = safe_ratio(gap_pre - gap_post, gap_pre)

            transitions.append({
                "run_number": run,
                "node_number": node,
                "node_depth": pre["node_depth"],
                "sep_round_node": r,
                "sep_round_run": pre["sep_round_run"],
                "sep_round_global": pre["sep_round_global"],
                "pre_lp_round_global": pre.get("lp_round_global"),
                "post_lp_round_global": post.get("lp_round_global"),
                "pre_lp_status": pre["lp_status"],
                "post_lp_status": post["lp_status"],
                "n_lp_rows_pre": pre["n_lp_rows"],
                "n_lp_rows_post": post["n_lp_rows"],
                "delta_n_lp_rows": numeric_delta(pre["n_lp_rows"], post["n_lp_rows"]),
                "n_lp_cols_pre": pre["n_lp_cols"],
                "n_lp_cols_post": post["n_lp_cols"],
                "delta_n_lp_cols": numeric_delta(pre["n_lp_cols"], post["n_lp_cols"]),
                "lp_obj_val_pre": pre["lp_obj_val"],
                "lp_obj_val_post": post["lp_obj_val"],
                "delta_lp_obj_val": lp_obj_delta,
                "lp_obj_improvement_ratio": lp_obj_ratio,
                "lp_iterations_total_pre": pre["lp_iterations_total"],
                "lp_iterations_total_post": post["lp_iterations_total"],
                "delta_lp_iterations_total": numeric_delta(pre["lp_iterations_total"], post["lp_iterations_total"]),
                "lp_iterations_node_pre": pre["lp_iterations_node"],
                "lp_iterations_node_post": post["lp_iterations_node"],
                "delta_lp_iterations_node": numeric_delta(pre["lp_iterations_node"], post["lp_iterations_node"]),
                "dual_bound_pre": pre["dual_bound"],
                "dual_bound_post": post["dual_bound"],
                "delta_dual_bound": numeric_delta(pre["dual_bound"], post["dual_bound"]),
                "primal_bound_pre": pre["primal_bound"],
                "primal_bound_post": post["primal_bound"],
                "delta_primal_bound": numeric_delta(pre["primal_bound"], post["primal_bound"]),
                "gap_pre": pre["gap"],
                "gap_post": post["gap"],
                "delta_gap": gap_delta,
                "relative_gap_improvement": relative_gap_improvement,
                "n_cuts_applied_pre": pre["n_cuts_applied"],
                "n_cuts_applied_post": post["n_cuts_applied"],
                "delta_n_cuts_applied": numeric_delta(pre["n_cuts_applied"], post["n_cuts_applied"]),
                "n_candidate_cuts": n_cands,
                "n_applied_cuts_round": n_applied_round,
                "n_cuts_generated_node_pre": pre["n_cuts_generated_node"],
                "n_cuts_generated_node_post": post["n_cuts_generated_node"],
                "delta_n_cuts_generated_node": numeric_delta(pre["n_cuts_generated_node"], post["n_cuts_generated_node"]),
                "n_open_nodes_pre": pre["n_open_nodes"],
                "n_open_nodes_post": post["n_open_nodes"],
                "delta_n_open_nodes": numeric_delta(pre["n_open_nodes"], post["n_open_nodes"]),
                "n_lp_solves_in_round": n_lp_solves,
            })

        self.sep_round_transitions = transitions

    def write(self, output_dir, summary=None):
        """将所有记录写入 CSV 与 JSON 摘要。"""
        return write_outputs(
            output_dir,
            self.candidate_records,
            self.applied_records,
            self.lp_records,
            sep_round_transitions=self.sep_round_transitions,
            summary=summary,
        )

    @staticmethod
    def _safe_float(value):
        try:
            v = float(value)
            if v == float("inf"):
                return "inf"
            if v == float("-inf"):
                return "-inf"
            return v
        except Exception:
            return None
