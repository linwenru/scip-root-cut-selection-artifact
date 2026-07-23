# SCIP 割平面日志记录器

使用 PySCIPOpt 监听 SCIP 求解 MIP 实例过程中的割平面生成与 LP 状态，并将结果导出为 CSV 文件。

## 功能

- 记录**候选割**（进入 cut selector 的所有割平面）
- 记录**被加入 LP 的割**（通过扫描 LP 中的 separator 行识别）
- 记录每次 LP 求解后的 **solver/LP 状态**
- 为每条割提取局部结构特征（非零系数、RHS、范数、最值、标准差、评分、排名等）

## 安装

```bash
pip install -r requirements.txt
```

要求：
- Python 3.10+
- SCIP 10.0+（与 PySCIPOpt 6.2+ 匹配）

## 使用

```bash
python main.py --input path/to/30n20b8.mps --output ./results --timelimit 300
```

参数说明：
- `--input`：输入 MPS 文件路径
- `--output`：输出目录（默认 `./output`）
- `--timelimit`：求解时间限制，秒（默认 `3600`）
- `--quiet`：关闭 SCIP 标准输出

## 输出文件

在输出目录中会生成以下 CSV：

| 文件 | 内容 |
|------|------|
| `candidate_cuts.csv` | 候选割的身份信息、局部结构特征、评分与排名 |
| `applied_cuts.csv` | 实际被加入 LP 的割的身份与结构特征 |
| `lp_states.csv` | 每次 LP 求解后的 solver/LP 状态 |
| `summary.json` | 求解摘要（状态、目标值、运行时间等） |
