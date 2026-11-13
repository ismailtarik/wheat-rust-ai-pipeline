#!/usr/bin/env python3
"""Group-aware 70/15/15 split for the wheat benchmark (reviewer request R2/R3).

Why: the original split is a random stratified split. 1,266 pHash near-duplicate
pairs cross train/val/test, and only the cross-CLASS ones were handled. This
script builds connected components over ALL near-duplicate pairs (same class or
not) and, optionally, over a filename-derived "origin image" key (to keep a
Mendeley original together with its augmentations). Whole groups are then
assigned to a single partition with StratifiedGroupKFold, so no near-duplicate
pair can cross partitions.

Usage (from benchmark/):
    python make_group_split.py \
        --manifest  ../data/processed/classification_v3/classification_manifest.csv \
        --pairs     ../data/reports/near_duplicate_audit/near_duplicate_pairs.csv \
        --out-dir   ../data/processed/classification_v4_group \
        --max-dist  6 --seed 42 \
        [--origin-regex '^(?P<origin>.+?)(?:_aug\\d+)?\\.[a-z]+$']

Outputs in --out-dir: classification_manifest_group.csv, group_split_report.json
Needs only pandas, numpy, scikit-learn (>=1.0).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 20                     # 14 / 3 / 3 folds -> 70 / 15 / 15 %
FOLDS_TRAIN, FOLDS_VAL = 14, 3


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cross_split_pairs(pairs, split_of):
    """Number of near-duplicate pairs whose two images lie in different partitions."""
    s1 = pairs["sha256_1"].map(split_of)
    s2 = pairs["sha256_2"].map(split_of)
    return int((s1 != s2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-dist", type=int, default=6, help="max pHash Hamming distance linking two images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--origin-regex", default=None,
                    help="regex with a named group 'origin' applied to the file name; "
                         "images with the same (class, origin) are forced into one group")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    old_col = "new_split" if "new_split" in df.columns else "split"
    for c in ("sha256", "canonical_class", old_col):
        assert c in df.columns, f"manifest lacks column {c!r}"
    assert df["sha256"].is_unique, "sha256 must be unique in the manifest"
    df = df.rename(columns={old_col: "split_old"})
    df = df.drop(columns=[c for c in ("split", "new_split") if c in df.columns])
    df["split_old"] = df["split_old"].astype(str).str.lower().replace({"validation": "val"})

    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs["hamming_distance"] <= a.max_dist]
    keep = set(df["sha256"])
    pairs = pairs[pairs["sha256_1"].isin(keep) & pairs["sha256_2"].isin(keep)].copy()

    old_map = dict(zip(df["sha256"], df["split_old"]))
    before = {
        "pairs_total": int(len(pairs)),
        "pairs_cross_split": cross_split_pairs(pairs, old_map),
        "pairs_cross_split_same_class": int(
            ((pairs["sha256_1"].map(old_map) != pairs["sha256_2"].map(old_map)) & pairs["same_class"]).sum()
        ) if "same_class" in pairs else None,
    }

    # 1) groups from pHash pairs (same-class AND cross-class)
    uf = UnionFind(df["sha256"])
    for s1, s2 in zip(pairs["sha256_1"], pairs["sha256_2"]):
        uf.union(s1, s2)

    # 2) groups from a filename-derived origin key (e.g. Mendeley augmentations)
    if a.origin_regex:
        rx = re.compile(a.origin_regex)
        path_col = "image_path" if "image_path" in df.columns else "destination_path"
        first = {}
        n_match = 0
        for sha, cls, p in zip(df["sha256"], df["canonical_class"], df[path_col]):
            m = rx.match(Path(str(p)).name)
            if not m:
                continue
            n_match += 1
            key = (cls, m.group("origin"))
            if key in first:
                uf.union(first[key], sha)
            else:
                first[key] = sha
        print(f"origin-regex matched {n_match} / {len(df)} file names")

    df["group_id"] = df["sha256"].map(uf.find)
    gsize = df.groupby("group_id")["sha256"].transform("size")
    df["group_size"] = gsize

    # a group spanning two classes cannot be stratified: keep it together, label by majority
    cls_per_group = df.groupby("group_id")["canonical_class"].nunique()
    mixed = cls_per_group[cls_per_group > 1].index
    strat = df.groupby("group_id")["canonical_class"].agg(lambda s: s.value_counts().idxmax())
    df["strat_label"] = df["group_id"].map(strat)

    # 3) stratified group split
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=a.seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(df, df["strat_label"], groups=df["group_id"])):
        fold[te] = k
    rng = np.random.RandomState(a.seed)
    order = rng.permutation(N_FOLDS)
    role = {}
    for i, f in enumerate(order):
        role[f] = "train" if i < FOLDS_TRAIN else ("val" if i < FOLDS_TRAIN + FOLDS_VAL else "test")
    df["split"] = [role[f] for f in fold]

    # 4) checks
    new_map = dict(zip(df["sha256"], df["split"]))
    after = {
        "pairs_total": int(len(pairs)),
        "pairs_cross_split": cross_split_pairs(pairs, new_map),
    }
    assert after["pairs_cross_split"] == 0, "leakage remains"
    assert (df.groupby("group_id")["split"].nunique() == 1).all()

    counts = df["split"].value_counts().to_dict()
    per_class = pd.crosstab(df["canonical_class"], df["split"])
    small = per_class[(per_class.get("val", 0) < 5) | (per_class.get("test", 0) < 5)].index.tolist()
    report = {
        "params": vars(a),
        "n_images": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "n_groups_with_more_than_1_image": int((df.groupby("group_id").size() > 1).sum()),
        "largest_group": int(df["group_size"].max()),
        "mixed_class_groups": int(len(mixed)),
        "split_sizes": counts,
        "split_fractions": {k: round(v / len(df), 4) for k, v in counts.items()},
        "leakage_before": before,
        "leakage_after": after,
        "images_that_changed_partition": int((df["split"] != df["split_old"]).sum()),
        "classes_with_<5_val_or_test_images": small,
        "per_class": per_class.to_dict(orient="index"),
    }

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df["new_split"] = df["split"]          # column name expected by train_eval.py
    df.drop(columns=["strat_label"]).to_csv(out / "classification_manifest_group.csv", index=False)
    (out / "group_split_report.json").write_text(json.dumps(report, indent=2, default=int))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2, default=int))


if __name__ == "__main__":
    main()
