"""根据 MIPLIB 2017 元数据生成训练/验证/测试划分。

只保留 status == easy 的实例，并按结构标签做多标签分层抽样，
确保各类结构标签（set_partitioning, set_covering, knapsack 等）
在 train/val/test 中都有较均匀分布。

用法：
    python prepare_split.py [--seed 42] [--train 0.7 --val 0.1 --test 0.2]

输出：
    - split/benchmark_tags.csv
    - split/instance_metadata.csv
    - split/train.test
    - split/val.test
    - split/test.test
"""

import argparse
import os
import sys
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd


MIPLIB_TAG_URL = "https://miplib.zib.de/tag_benchmark.html"
STRUCTURAL_TAGS = {
    "mixed_binary",
    "variable_bound",
    "precedence",
    "set_partitioning",
    "general_linear",
    "invariant_knapsack",
    "aggregations",
    "set_packing",
    "decomposition",
    "cardinality",
    "set_covering",
    "binary",
    "knapsack",
    "binpacking",
    "integer_knapsack",
    "equation_knapsack",
}
IGNORED_TAGS = {"benchmark", "benchmark_suitable", "feasibility", "infeasible"}


def parse_tags_from_html(html_path):
    """解析 MIPLIB tag_benchmark.html 中的标签表格。"""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Parsing MIPLIB HTML requires beautifulsoup4. "
            "Install it or provide a pre-built tags CSV."
        ) from exc

    with open(html_path) as f:
        soup = BeautifulSoup(f, "html.parser")
    table = soup.find("table", {"id": "miplibtable"})
    if table is None:
        raise ValueError("Could not find table with id='miplibtable'")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 12:
            continue
        name = tds[0].get_text(strip=True) + ".mps.gz"
        status = tds[1].get_text(strip=True)
        vars_ = tds[2].get_text(strip=True)
        bins = tds[3].get_text(strip=True)
        ints = tds[4].get_text(strip=True)
        conts = tds[5].get_text(strip=True)
        constr = tds[6].get_text(strip=True)
        nonz = tds[7].get_text(strip=True)
        tags = " ".join(span.get_text(strip=True) for span in tds[11].find_all("span"))
        rows.append([name, status, vars_, bins, ints, conts, constr, nonz, tags])
    return pd.DataFrame(
        rows,
        columns=[
            "instance_name",
            "status",
            "vars",
            "bin_vars",
            "int_vars",
            "cont_vars",
            "constr",
            "nonzeroes",
            "tags",
        ],
    )


def ensure_tags_csv(tags_path):
    """如果 tags CSV 不存在，下载并解析 MIPLIB HTML。"""
    if os.path.exists(tags_path):
        return pd.read_csv(tags_path)

    os.makedirs(os.path.dirname(tags_path), exist_ok=True)
    html_path = tags_path.replace(".csv", ".html")
    if not os.path.exists(html_path):
        print(f"Downloading {MIPLIB_TAG_URL} ...")
        urllib.request.urlretrieve(MIPLIB_TAG_URL, html_path)
    df = parse_tags_from_html(html_path)
    df.to_csv(tags_path, index=False)
    return df


def iterative_multilabel_split(df, tags_series, train_size, val_size, test_size, seed):
    """简化的多标签迭代分层抽样。"""
    rng = np.random.default_rng(seed)
    n = len(df)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val

    targets = {"train": n_train, "val": n_val, "test": n_test}
    assigned = {"train": set(), "val": set(), "test": set()}

    # 标签 -> 包含该标签的实例索引
    tag_to_indices = {}
    for idx, tags in tags_series.items():
        for tag in tags:
            tag_to_indices.setdefault(tag, set()).add(idx)

    # 按标签覆盖实例数从少到多处理，稀有标签优先保证均匀
    sorted_tags = sorted(tag_to_indices, key=lambda t: len(tag_to_indices[t]))
    for tag in sorted_tags:
        candidates = list(
            tag_to_indices[tag] - assigned["train"] - assigned["val"] - assigned["test"]
        )
        rng.shuffle(candidates)
        for idx in candidates:
            ratios = {}
            for split in assigned:
                if targets[split] == 0:
                    ratios[split] = float("inf")
                else:
                    count = sum(1 for j in assigned[split] if tag in tags_series[j])
                    ratios[split] = count / targets[split]
            best = min(assigned, key=lambda s: ratios[s])
            if len(assigned[best]) < targets[best]:
                assigned[best].add(idx)

    # 分配剩余实例
    remaining = [idx for idx in df.index if idx not in assigned["train"] | assigned["val"] | assigned["test"]]
    rng.shuffle(remaining)
    for split in ["train", "val", "test"]:
        needed = targets[split] - len(assigned[split])
        for idx in remaining[:needed]:
            assigned[split].add(idx)
        remaining = remaining[needed:]

    return (
        df.loc[sorted(assigned["train"])].copy(),
        df.loc[sorted(assigned["val"])].copy(),
        df.loc[sorted(assigned["test"])].copy(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split for MIPLIB 2017 easy benchmark instances."
    )
    parser.add_argument(
        "--tags-file",
        default="split/benchmark_tags.csv",
        help="CSV with instance tags (downloaded/parsed if missing).",
    )
    parser.add_argument(
        "--output-dir",
        default="split",
        help="Directory to write split .test files and metadata.",
    )
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print("Error: train + val + test must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    df = ensure_tags_csv(args.tags_file)

    # 只保留 easy 实例
    df = df[df["status"] == "easy"].copy()

    # 解析结构标签
    df["tag_list"] = df["tags"].apply(
        lambda s: [t for t in str(s).split() if t in STRUCTURAL_TAGS]
    )

    # 多标签分层抽样
    train, val, test = iterative_multilabel_split(
        df,
        df["tag_list"],
        args.train,
        args.val,
        args.test,
        args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output_dir, f"{split_name}.test")
        with open(path, "w") as f:
            for name in sorted(split_df["instance_name"].tolist()):
                f.write(name + "\n")

    # 保存元数据
    metadata_path = os.path.join(args.output_dir, "instance_metadata.csv")
    df[["instance_name", "status", "vars", "bin_vars", "int_vars", "cont_vars", "constr", "nonzeroes", "tags"]].to_csv(
        metadata_path, index=False
    )

    print(f"Total easy instances: {len(df)}")
    print(f"\nSplit sizes:")
    print(f"  train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"  test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

    # 打印各集合的标签分布
    print("\nStructural tag distribution across splits:")
    for tag in sorted(STRUCTURAL_TAGS):
        total = sum(1 for tags in df["tag_list"] if tag in tags)
        if total == 0:
            continue
        train_n = sum(1 for tags in train["tag_list"] if tag in tags)
        val_n = sum(1 for tags in val["tag_list"] if tag in tags)
        test_n = sum(1 for tags in test["tag_list"] if tag in tags)
        print(
            f"  {tag:22s}: total={total:3d}  "
            f"train={train_n}/{len(train)} ({train_n/len(train)*100:5.1f}%)  "
            f"val={val_n}/{len(val)} ({val_n/len(val)*100:5.1f}%)  "
            f"test={test_n}/{len(test)} ({test_n/len(test)*100:5.1f}%)"
        )

    print(f"\nSplit files written to {args.output_dir}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
