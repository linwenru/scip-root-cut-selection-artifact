"""PySCIPOpt 回调：Cutsel 与 Eventhdlr。"""

from pyscipopt import SCIP_RESULT, SCIP_EVENTTYPE, SCIP_LPSOLSTAT, SCIP_ROWORIGINTYPE, SCIP_STAGE
from pyscipopt.scip import Cutsel, Eventhdlr

from .row_features import extract_row_features, compute_hybrid_score, _get_cut_id, _origin_type_name, _is_cut_origin_type


class CutselCallback(Cutsel):
    """自定义割平面选择器：复现 SCIP 默认 hybrid 选择器逻辑并记录候选割。"""

    def __init__(self, state_tracker, candidate_records, sep_round_pre_states):
        super().__init__()
        self.state_tracker = state_tracker
        self.candidate_records = candidate_records
        self.sep_round_pre_states = sep_round_pre_states
        self._selection_count = 0

    def cutselinit(self):
        pass

    def _get_hybrid_params(self, root):
        """读取 SCIP 默认 hybrid cut selector 的参数。"""
        model = self.model
        return {
            "efficacyweight": float(model.getParam("cutselection/hybrid/efficacyweight")),
            "dircutoffdistweight": float(model.getParam("cutselection/hybrid/dircutoffdistweight")),
            "objparalweight": float(model.getParam("cutselection/hybrid/objparalweight")),
            "intsupportweight": float(model.getParam("cutselection/hybrid/intsupportweight")),
            "minortho": float(
                model.getParam("cutselection/hybrid/minorthoroot" if root else "cutselection/hybrid/minortho")
            ),
            "goodscorefac": 0.9,
            "badscorefac": 0.0,
        }

    @staticmethod
    def _compute_hybrid_scores(model, cuts, sol, params):
        """计算一组割的 hybrid score。"""
        return [compute_hybrid_score(model, cut, sol=sol, params=params) for cut in cuts]

    @staticmethod
    def _filter_by_parallelism(model, cut, cuts, scores, ncuts, goodscore, goodmaxparall, maxparall):
        """依据正交性过滤与给定割过于平行的割。"""
        i = ncuts - 1
        while i >= 0:
            try:
                parall = model.getRowParallelism(cut, cuts[i], 101)
            except Exception:
                parall = 0.0
            thismaxparall = goodmaxparall if scores[i] >= goodscore else maxparall
            if parall > thismaxparall:
                ncuts -= 1
                cuts[i], cuts[ncuts] = cuts[ncuts], cuts[i]
                scores[i], scores[ncuts] = scores[ncuts], scores[i]
            i -= 1
        return ncuts

    def _hybrid_select(self, cuts, scores, forcedcuts, forcedscores, maxnselectedcuts, params):
        """执行 hybrid 选择算法：forced cuts 过滤 + 贪心选择。"""
        model = self.model
        maxscore = 0.0
        if scores:
            maxscore = max(maxscore, max(scores))
        if forcedscores:
            maxscore = max(maxscore, max(forcedscores))

        goodscore = maxscore * params["goodscorefac"]
        badscore = maxscore * params["badscorefac"]
        maxparall = 1.0 - params["minortho"]
        goodmaxparall = max(0.5, 1.0 - params["minortho"])

        # 复制用于选择过程，避免破坏原始顺序
        sel_cuts = list(cuts)
        sel_scores = list(scores)
        ncuts = len(sel_cuts)

        # forced cuts 用于过滤候选割
        for forced_cut, _ in zip(forcedcuts, forcedscores):
            ncuts = self._filter_by_parallelism(
                model, forced_cut, sel_cuts, sel_scores, ncuts,
                goodscore, goodmaxparall, maxparall,
            )

        # 贪心选择
        selected_flags = [False] * len(cuts)
        cut_to_idx = {id(cut): i for i, cut in enumerate(cuts)}
        n_selected = 0
        while ncuts > 0:
            best_idx = max(range(ncuts), key=lambda i: sel_scores[i])
            sel_cuts[0], sel_cuts[best_idx] = sel_cuts[best_idx], sel_cuts[0]
            sel_scores[0], sel_scores[best_idx] = sel_scores[best_idx], sel_scores[0]

            if sel_scores[0] < badscore:
                break

            orig_idx = cut_to_idx.get(id(sel_cuts[0]))
            if orig_idx is not None:
                selected_flags[orig_idx] = True
            n_selected += 1

            if n_selected == maxnselectedcuts:
                break

            selected_cut = sel_cuts[0]
            sel_cuts = sel_cuts[1:]
            sel_scores = sel_scores[1:]
            ncuts -= 1
            ncuts = self._filter_by_parallelism(
                model, selected_cut, sel_cuts, sel_scores, ncuts,
                goodscore, goodmaxparall, maxparall,
            )

        return selected_flags, n_selected

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """记录候选割并返回 DIDNOTFIND，让 SCIP 调用默认选择器。

        注意：自定义 cut selector 即使返回 DIDNOTFIND，SCIP 也未必会回退到
        默认 hybrid selector（实测 hybrid Calls 仍为 0）。这里按网上描述的方式
        实现，用于验证求解轨迹是否与裸跑 SCIP 一致。
        """
        model = self.model
        self.state_tracker.increment_sep_round()
        self._record_pre_sep_state(model)

        params = self._get_hybrid_params(root)
        try:
            sol = model.getBestSol()
        except Exception:
            sol = None

        # 计算所有候选割与 forced cuts 的 hybrid score（用于特征记录）
        cut_scores = self._compute_hybrid_scores(model, cuts, sol, params)
        forced_scores = self._compute_hybrid_scores(model, forcedcuts, sol, params)

        # 按 hybrid score 对候选割排序并记录（不实际选择）
        ranked = sorted(
            zip(range(len(cuts)), cut_scores, cuts),
            key=lambda x: -x[1],
        )
        for rank, (orig_idx, score, cut) in enumerate(ranked, start=1):
            features = extract_row_features(model, cut)
            record = {
                **self.state_tracker.to_dict(),
                "root": bool(root),
                "is_forced": False,
                "is_selected": None,  # 由默认选择器决定，Python 层未知
                "original_index": int(orig_idx),
                "rank": int(rank),
                "score": float(score),
                **features,
            }
            self.candidate_records.append(record)

        # 记录 forced cuts（必须加入 LP 的割）
        for cut, score in zip(forcedcuts, forced_scores):
            features = extract_row_features(model, cut)
            record = {
                **self.state_tracker.to_dict(),
                "root": bool(root),
                "is_forced": True,
                "is_selected": True,
                "original_index": -1,
                "rank": 0,
                "score": float(score),
                **features,
            }
            self.candidate_records.append(record)

        self._selection_count += 1

        return {
            "nselectedcuts": 0,
            "result": SCIP_RESULT.DIDNOTFIND,
        }

    def _record_pre_sep_state(self, model):
        """在每次分离轮次开始时记录当前 LP 状态（作为该轮的前状态）。

        只在 SOLVING 阶段记录；presolve 等阶段部分 LP 状态 API 不可用，
        出现异常时跳过该次记录，避免中断 SCIP 的割平面流程。
        """
        if model.getStage() != SCIP_STAGE.SOLVING:
            return
        try:
            lp_status = model.getLPSolstat()
        except Exception:
            lp_status = None
        try:
            n_cuts_applied = int(model.getNCutsApplied())
        except Exception:
            # 在 presolving 等阶段 getNCutsApplied 可能不可用
            n_cuts_applied = None
        try:
            record = {
                **self.state_tracker.to_dict(),
                "n_lp_rows": int(model.getNLPRows()),
                "n_lp_cols": int(model.getNLPCols()),
                "lp_status": EventCallback._lp_status_name(lp_status),
                "lp_obj_val": EventCallback._safe_float(model.getLPObjVal()),
                "lp_iterations_total": int(model.getNLPIterations()),
                "lp_iterations_node": int(model.getNNodeLPIterations()),
                "dual_bound": EventCallback._safe_float(model.getDualbound()),
                "primal_bound": EventCallback._safe_float(model.getPrimalbound()),
                "gap": EventCallback._safe_float(model.getGap()),
                "n_cuts_applied": n_cuts_applied,
                "n_open_nodes": EventCallback._count_open_nodes(model),
            }
            self.sep_round_pre_states.append(record)
        except Exception:
            # 某些 LP 状态 API 在特殊阶段不可用，跳过该次快照
            pass


class EventCallback(Eventhdlr):
    """事件处理器：记录 LP 状态与当前 LP 中新加入的割平面行。"""

    def __init__(self, state_tracker, applied_records, lp_records):
        super().__init__()
        self.state_tracker = state_tracker
        self.state_tracker.set_on_restart(self._reset_run_counters)
        self.applied_records = applied_records
        self.lp_records = lp_records
        self._current_lp_rows = set()
        self._n_cuts_generated_global = 0
        self._n_cuts_generated_run = 0

    def eventinit(self):
        """注册关注的事件。"""
        self.model.catchEvent(
            SCIP_EVENTTYPE.NODEFOCUSED
            | SCIP_EVENTTYPE.LPSOLVED
            | SCIP_EVENTTYPE.ROWADDEDSEPA
            | SCIP_EVENTTYPE.ROWADDEDLP,
            self,
        )

    def eventexit(self):
        pass

    def eventexec(self, event):
        try:
            self._eventexec_internal(event)
        except Exception:
            # 事件处理器中的异常不应中断 SCIP 求解过程。
            pass

    def _eventexec_internal(self, event):
        event_type = event.getType()
        model = self.model

        if event_type == SCIP_EVENTTYPE.NODEFOCUSED:
            node = event.getNode()
            last_node = self.state_tracker.node_number
            self.state_tracker.update_node(node)
            if self.state_tracker.node_number != last_node:
                pass

        elif event_type == SCIP_EVENTTYPE.ROWADDEDLP:
            self._record_row_added_lp(model, event)

        elif event_type == SCIP_EVENTTYPE.ROWADDEDSEPA:
            self._record_generated_cut(model, event)

        elif event_type == SCIP_EVENTTYPE.LPSOLVED:
            self.state_tracker.increment_lp_round()
            self._record_lp_state(model)
            # self._scan_lp_cuts(model)

    def _reset_run_counters(self):
        """SCIP restart 后重置 run 级计数器。LP 行集合也清空，因为 LP 已重新构建。"""
        self._n_cuts_generated_run = 0
        self._current_lp_rows = set()

    @staticmethod
    def _lp_status_name(lp_status):
        """将 SCIP_LPSOLSTAT 值转换为可读名称。"""
        if not isinstance(lp_status, int):
            return getattr(lp_status, "name", str(lp_status))
        for name in ["NOTSOLVED", "OPTIMAL", "INFEASIBLE", "UNBOUNDEDRAY", "OBJLIMIT", "ITERLIMIT", "TIMELIMIT", "ERROR"]:
            if getattr(SCIP_LPSOLSTAT, name, None) == lp_status:
                return name
        return str(lp_status)

    @staticmethod
    def _count_open_nodes(model):
        """getOpenNodes() 返回 (leaves, children, siblings) 三个 Node 列表，求和得到开放节点总数。"""
        try:
            leaves, children, siblings = model.getOpenNodes()
            return len(leaves) + len(children) + len(siblings)
        except Exception:
            return 0

    def _record_generated_cut(self, model, event):
        """ROWADDEDSEPA 事件触发时，累计 separator 加入 storage 的割。"""
        try:
            row = event.getRow()
            origin_type = row.getOrigintype()
            if _is_cut_origin_type(origin_type) or _is_cons_origin_type(origin_type):
                self._n_cuts_generated_global += 1
                self._n_cuts_generated_run += 1
                self.state_tracker.n_cuts_generated_node += 1
        except Exception:
            pass

    def _record_lp_state(self, model):
        """记录当前 solver / LP 状态。每次 SCIP_EVENTTYPE.LPSOLVED 会调用"""
        if model.getStage() != SCIP_STAGE.SOLVING:
            return
        lp_status = model.getLPSolstat()
        record = {
            **self.state_tracker.to_dict(),
            "n_lp_rows": int(model.getNLPRows()), # Retrieve the number of rows currently in the LP.
            "n_lp_cols": int(model.getNLPCols()), # Retrieve the number of columns currently in the LP.
            "lp_status": self._lp_status_name(lp_status),
            "lp_obj_val": self._safe_float(model.getLPObjVal()), # Gets objective value of current LP (which is the sum of column and loose objective value).
            "lp_iterations": int(model.lpiGetIterations()), # Get the iteration count of the last solved LP.
            "lp_iterations_total": int(model.getNLPIterations()), # Returns the total number of LP iterations so far. 跨 run 累计。
            "lp_iterations_node": int(model.getNNodeLPIterations()), # Gets number of LP iterations used for solving node relaxations so far. 跨 run 累计。
            "lp_tierations_strongbranch": int(model.getNStrongbranchLPIterations()), # Gets number of LP iterations used for strong branching so far. 跨 run 累计。
            "dual_bound": self._safe_float(model.getDualbound()), # Retrieve the best dual bound.
            "primal_bound": self._safe_float(model.getPrimalbound()), # Retrieve the best primal bound.
            "gap": self._safe_float(model.getGap()), # Retrieve the gap, i.e. abs((primalbound - dualbound)/min(abs(primalbound),abs(dualbound)))
            "n_processed_nodes_global": int(model.getNTotalNodes()), # Gets number of processed nodes in all runs, including the focus node.
            "n_processed_nodes_run": int(model.getNNodes()), # Gets number of processed nodes in current run, including the focus node.
            "n_open_nodes": self._count_open_nodes(model), # model.getOpenNodes() returns (leaves, children, siblings), count total.
            "n_cuts_generated_global": self._n_cuts_generated_global, # Cumulative cuts added to separation storage via ROWADDEDSEPA (all runs).
            "n_cuts_generated_run": self._n_cuts_generated_run, # Cumulative cuts added in current run; reset on restart.
            "n_cuts_generated_node": self.state_tracker.n_cuts_generated_node, # Cumulative cuts added in current node; reset on node change.
            "n_cuts_applied": int(model.getNCutsApplied()), # Retrieve number of currently applied cuts.
            "solving_time": self._safe_float(model.getSolvingTime()), # Retrieve the current solving time in seconds.
        }
        self.lp_records.append(record)

    def _scan_lp_cuts(self, model):
        """扫描当前 LP 中的割平面行，识别本轮新加入 LP 的割。每次 SCIP_EVENTTYPE.LPSOLVED 会调用"""
        try:
            rows = model.getLPRowsData()
        except Exception:
            return

        if rows is None:
            return

        current_ids = set()
        for row in rows:
            try:
                origin_type = row.getOrigintype()
                if not _is_cut_origin_type(origin_type):
                    continue
                cut_id = _get_cut_id(row)
                current_ids.add(cut_id)
                if cut_id in self._current_lp_rows: #（上一次 LP 扫描留下的集合 + 本次扫描新加入的 cut_id ）
                    continue
                self._record_applied_cut(model, row) # 执行这行会往 _current_lp_rows 中加入 cut_id
            except Exception:
                continue

        self._current_lp_rows = current_ids # 本次 LP 扫描到的所有 cut_id 赋值给 _current_lp_rows ，为下一次扫描做准备

    def _record_row_added_lp(self, model, event):
        """ROWADDEDLP 事件触发时，记录被加入 LP 的割平面行。

        ROWADDEDLP 只在新行被加入 LP 时触发，因此出现在这里的 CONS 行不是
        LP 初始化时就存在的原始约束，而是被 SCIP 作为割加入 LP 的原始约束。
        """
        try:
            row = event.getRow()
            origin_type = row.getOrigintype()
            if not (_is_cut_origin_type(origin_type) or self._is_cons_origin_type(origin_type)):
                return
            # cut_id = _get_cut_id(row)
            # if cut_id in self._current_lp_rows:
            #     return
            features = extract_row_features(model, row)
            record = {
                **self.state_tracker.to_dict(),
                **features,
            }
            self.applied_records.append(record)
            # self._current_lp_rows.add(cut_id)
        except Exception:
            pass

    @staticmethod
    def _is_cons_origin_type(origin_type_val):
        """判断行来源是否为 CONS（原始约束）。

        在 ROWADDEDLP 事件中使用：CONS 出现在 ROWADDEDLP 中时，说明该原始约束
        被 SCIP 作为割加入 LP，应记录为 applied cut。
        """
        if not isinstance(origin_type_val, int):
            name = getattr(origin_type_val, "name", str(origin_type_val))
            return name == "CONS"
        return origin_type_val == SCIP_ROWORIGINTYPE.CONS

    def _record_applied_cut(self, model, row):
        """记录一条被加入 LP 的割平面行；若已记录过则去重跳过。"""
        try:
            origin_type = row.getOrigintype()
            if not _is_cut_origin_type(origin_type):
                return
        except Exception:
            return
        cut_id = _get_cut_id(row)
        if cut_id in self._current_lp_rows:
            return
        features = extract_row_features(model, row)
        record = {
            **self.state_tracker.to_dict(),
            **features,
        }
        self.applied_records.append(record)
        self._current_lp_rows.add(cut_id)

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
