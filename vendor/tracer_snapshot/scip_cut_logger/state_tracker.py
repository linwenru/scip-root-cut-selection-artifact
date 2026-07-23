"""跟踪 SCIP 求解过程中的节点、分离轮次与 LP 求解轮次。"""


class StateTracker:
    """维护求解过程中的全局/局部计数器。"""

    def __init__(self, on_restart=None):
        self.run_number = 1
        self.node_number = -1
        self.node_depth = 0
        self.sep_round_node = 0
        self.sep_round_run = 0
        self.sep_round_global = 0
        self.lp_round_node = 0
        self.lp_round_run = 0
        self.lp_round_global = 0
        self.n_cuts_generated_node = 0
        self._last_node_number = -1
        self._has_seen_node_in_current_run = False
        self._on_restart = on_restart

    def set_on_restart(self, callback):
        """设置 restart 检测后的回调函数。"""
        self._on_restart = callback

    def update_node(self, node):
        """切换到新节点时重置节点级计数器，并检测 SCIP restart。"""
        if node is None:
            return
        new_number = node.getNumber()
        new_depth = node.getDepth()

        # Restart 检测：SCIP 在 restart 时会丢弃整棵 B&B 树，新根节点深度回到 0。
        # 只要当前 run 已经处理过任意节点（包括根节点），再次遇到深度为 0 的节点就说明发生了 restart。
        # 注意：即使新根节点的编号与上一 run 相同（例如仍是 1），也必须识别为新的 run。
        if new_depth == 0 and self._has_seen_node_in_current_run:
            self.run_number += 1
            self.sep_round_run = 0
            self.lp_round_run = 0
            self._has_seen_node_in_current_run = False
            self._last_node_number = -1  # 新 run 的根节点编号可能仍是 1，需重置以重新识别
            if self._on_restart is not None:
                self._on_restart()

        # 同一节点再次触发时跳过（更新节点的计数器只需执行一次）。
        if new_number == self._last_node_number:
            return

        self.node_number = new_number
        self.node_depth = new_depth
        self.sep_round_node = 0
        self.lp_round_node = 0
        self.n_cuts_generated_node = 0
        self._last_node_number = new_number
        self._has_seen_node_in_current_run = True

    def increment_sep_round(self):
        """cut selector 被调用一次视为一次分离轮次。"""
        self.sep_round_node += 1
        self.sep_round_run += 1
        self.sep_round_global += 1
        return self.sep_round_node, self.sep_round_run, self.sep_round_global

    def increment_lp_round(self):
        """每次 LP 求解完成视为一次 LP 轮次。"""
        self.lp_round_node += 1
        self.lp_round_run += 1
        self.lp_round_global += 1
        return self.lp_round_node, self.lp_round_run, self.lp_round_global

    def to_dict(self):
        return {
            "run_number": self.run_number,
            "node_number": self.node_number,
            "node_depth": self.node_depth,
            "sep_round_node": self.sep_round_node,
            "sep_round_run": self.sep_round_run,
            "sep_round_global": self.sep_round_global,
            "lp_round_node": self.lp_round_node,
            "lp_round_run": self.lp_round_run,
            "lp_round_global": self.lp_round_global,
            "n_cuts_generated_node": self.n_cuts_generated_node,
        }
