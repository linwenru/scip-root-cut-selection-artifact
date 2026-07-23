"""提取 SCIP Row 的局部结构特征。"""

import numpy as np

from pyscipopt import SCIP_ROWORIGINTYPE


def _origin_type_name(origin_type_val):
    """将行来源类型值安全转换为名称字符串。"""
    if not isinstance(origin_type_val, int):
        return getattr(origin_type_val, "name", str(origin_type_val))
    # SCIP C API 中 CONSHDLR = 1，但部分 PySCIPOpt 版本未暴露该枚举名，
    # 因此先显式处理数值 1，再回退到 PySCIPOpt 暴露的枚举名。
    if origin_type_val == 1:
        return "CONSHDLR"
    for name in ["UNSPEC", "CONS", "SEPA", "REOPT"]:
        if getattr(SCIP_ROWORIGINTYPE, name, None) == origin_type_val:
            return name
    return str(origin_type_val)


def _is_cut_origin_type(origin_type_val):
    """判断行来源是否为割平面（SEPA 或 CONSHDLR）。

    SCIP 中割平面不仅由 separator 产生，也可以由 constraint handler 的
    separation 回调产生。原始约束（CONS）与再优化（REOPT）行不属于割平面。
    """
    if not isinstance(origin_type_val, int):
        name = getattr(origin_type_val, "name", str(origin_type_val))
        return name in ("SEPA", "CONSHDLR")
    return origin_type_val in (1, SCIP_ROWORIGINTYPE.SEPA)


def _column_var_name(col):
    """从 Column 对象安全获取变量名。"""
    try:
        var = col.getVar()
        return var.name if hasattr(var, "name") else str(var)
    except Exception:
        return str(col)


def _get_cut_id(row):
    """基于割的内容与名称生成稳定 ID。

    同时纳入 row.name，避免两个不同来源但系数/边界完全相同的行
    （例如 mcseq0 与 ewseq0 约束）产生相同的 cut_id。
    """
    cols = row.getCols()
    vals = row.getVals()
    rhs = row.getRhs()
    lhs = row.getLhs()
    constant = row.getConstant()
    name = getattr(row, "name", None) or ""

    items = tuple(sorted((_column_var_name(col), float(v)) for col, v in zip(cols, vals)))
    key = (name,) + items + (float(rhs), float(lhs), float(constant))
    return hash(key) & 0xFFFFFFFF


def extract_row_features(model, row):
    """提取一条割/行的局部结构特征。

    Returns
    -------
    dict
        包含非零系数个数、RHS、LHS、常数项、范数、最大/最小绝对值系数、
        绝对值系数标准差、整数变量个数、来源类型、 efficacy 等指标。
    """
    try:
        cols = row.getCols()
        vals = row.getVals()
        rhs = row.getRhs()
        lhs = row.getLhs()
        constant = row.getConstant()
    except Exception:
        return {"cut_id": None, "cut_name": getattr(row, "name", None), "origin_type": None}

    coeffs = np.array(vals, dtype=float)
    nnz = int(len(coeffs))

    origin_type_name = _origin_type_name(row.getOrigintype())

    features = {
        "cut_id": _get_cut_id(row),
        "cut_name": row.name,
        "origin_type": origin_type_name,
        "nnz": nnz,
        "rhs": float(rhs),
        "lhs": float(lhs),
        "constant": float(constant),
        "is_local": bool(row.isLocal()),
        "is_modifiable": bool(row.isModifiable()),
        "is_removable": bool(row.isRemovable()),
        "is_integral": bool(row.isIntegral()),
        "in_global_cutpool": bool(row.isInGlobalCutpool()),
        "lp_position": int(row.getLPPos()),
    }

    if nnz > 0:
        abs_coeffs = np.abs(coeffs)
        features["coeff_norm_l2"] = float(np.linalg.norm(coeffs))
        features["coeff_norm_l1"] = float(np.sum(abs_coeffs))
        features["coeff_max_abs"] = float(np.max(abs_coeffs))
        features["coeff_min_abs"] = float(np.min(abs_coeffs))
        features["coeff_mean_abs"] = float(np.mean(abs_coeffs))
        features["coeff_std_abs"] = float(np.std(abs_coeffs))
        features["coeff_sparsity_ratio"] = float(nnz / len(cols)) if cols else 0.0
    else:
        features["coeff_norm_l2"] = 0.0
        features["coeff_norm_l1"] = 0.0
        features["coeff_max_abs"] = 0.0
        features["coeff_min_abs"] = 0.0
        features["coeff_mean_abs"] = 0.0
        features["coeff_std_abs"] = 0.0
        features["coeff_sparsity_ratio"] = 0.0

    # SCIP 内置评分
    try:
        features["efficacy"] = float(model.getCutEfficacy(row))
    except Exception:
        features["efficacy"] = None

    try:
        features["obj_parallelism"] = float(model.getRowObjParallelism(row))
    except Exception:
        features["obj_parallelism"] = None

    try:
        sol = model.getBestSol()
        features["cutoff_distance"] = float(model.getCutLPSolCutoffDistance(row, sol))
    except Exception:
        features["cutoff_distance"] = None

    try:
        features["n_int_cols"] = int(model.getRowNumIntCols(row))
    except Exception:
        features["n_int_cols"] = None

    return features


def compute_hybrid_score(model, row, sol=None, params=None):
    """计算 SCIP 默认 hybrid cut selector 的割评分。

    复现 SCIP 中 cutsel_hybrid.c 的 scoring() 函数：
    score = efficacy_weight * efficacy
          + objparal_weight * obj_parallelism
          + intsupport_weight * n_int_cols / nnz
          + dircutoffdist_weight * directed_cutoff_distance
          + 1e-4 (if in global cutpool)

    Parameters
    ----------
    model : pyscipopt.Model
        当前 SCIP 模型。
    row : pyscipopt.Row
        待评分的割/行。
    sol : pyscipopt.Solution or None, optional
        当前最优解，用于计算 directed cutoff distance。为 None 时不使用。
    params : dict or None, optional
        hybrid 选择器参数，默认与 SCIP 默认值一致。

    Returns
    -------
    float
        hybrid 评分。
    """
    if params is None:
        params = {
            "efficacyweight": 1.0,
            "dircutoffdistweight": 0.0,
            "objparalweight": 0.1,
            "intsupportweight": 0.1,
        }

    efficacy_weight = params.get("efficacyweight", 1.0)
    dircutoffdist_weight = params.get("dircutoffdistweight", 0.0)
    objparal_weight = params.get("objparalweight", 0.1)
    intsupport_weight = params.get("intsupportweight", 0.1)

    efficacy = 0.0
    if efficacy_weight != 0.0 or dircutoffdist_weight != 0.0:
        try:
            efficacy = float(model.getCutEfficacy(row))
        except Exception:
            efficacy = 0.0

    objparal = 0.0
    if objparal_weight != 0.0:
        try:
            objparal = float(model.getRowObjParallelism(row))
        except Exception:
            objparal = 0.0

    intsupport = 0.0
    if intsupport_weight != 0.0:
        try:
            nnz = int(row.getNNonz())
            if nnz > 0:
                intsupport = intsupport_weight * int(model.getRowNumIntCols(row)) / nnz
        except Exception:
            intsupport = 0.0

    if sol is not None and dircutoffdist_weight > 0.0:
        if row.isLocal():
            dircutoff = efficacy
        else:
            try:
                cutoff_dist = float(model.getCutLPSolCutoffDistance(row, sol))
            except Exception:
                cutoff_dist = 0.0
            dircutoff = max(cutoff_dist, efficacy)
        score = dircutoffdist_weight * dircutoff
    else:
        efficacy_weight += dircutoffdist_weight
        score = 0.0

    score += objparal_weight * objparal + intsupport + efficacy_weight * efficacy

    if row.isInGlobalCutpool():
        score += 1e-4

    return score
