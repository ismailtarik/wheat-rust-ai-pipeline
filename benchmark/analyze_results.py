"""Analyse des résultats du benchmark (à lancer sur Colab ou en local, sans GPU).

Usage :
    python analyze_results.py /content/drive/MyDrive/wheat_attention_v2 ResNet50 ResNet50_TAMv2_after_layer4

Arguments : dossier de sortie du benchmark, modèle de référence, modèle à comparer (noms de dossiers),
et (optionnel) liste de seeds à retenir, ex. 42,123,2024 (recommandé : même seeds pour tous les modèles).
Lit, pour chaque seed, <dossier>/<modèle>/seed_N/test_evaluation/{test_predictions.csv, classification_report.csv}.
Produit : tableau moyenne ± écart-type de tous les modèles, F1 par classe (réf. vs comparé), McNemar par seed.

Option --clean=MANIFEST,DUPCSV[,SEUIL] : évalue uniquement sur les images de test SANS quasi-doublon dans le train
(SEUIL = distance de Hamming, 6 par défaut). Exemple :
    python analyze_results.py DOSSIER ResNet50 ResNet50_TAMv2_after_layer4 42,123,2024 \
        --clean=/content/wheat-rust-ai-pipeline/data/processed/classification_v3/classification_manifest.csv,DOSSIER/duplicates_test_vs_train.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

CLEAN_MASK, SUFFIX = None, ""
for _a in list(sys.argv[1:]):
    if _a.startswith("--clean="):
        sys.argv.remove(_a)
        _p = _a.split("=", 1)[1].split(",")
        _man = pd.read_csv(_p[0])
        _man = _man[_man.new_split.astype(str) == "test"].reset_index(drop=True)
        _dup = pd.read_csv(_p[1])
        _thr = int(_p[2]) if len(_p) > 2 else 6
        _near = set(_dup.loc[_dup.distance <= _thr, "v3_destination_path"].astype(str))
        CLEAN_MASK = ~_man.v3_destination_path.astype(str).isin(_near).values
        SUFFIX = "_clean"
        print(f"[--clean] test restreint à {int(CLEAN_MASK.sum())} / {len(CLEAN_MASK)} images sans quasi-doublon (distance > {_thr}).")

_bad = [x for x in sys.argv[1:] if x.startswith("--")]
if _bad:
    sys.exit(f"Option non reconnue : {_bad} (écrire --clean=MANIFEST,DUPCSV sans espace autour du '=').")
print("analyze_results (version avec --clean) - mode :", "test SANS quasi-doublons" if CLEAN_MASK is not None else "test complet")

base = Path(sys.argv[1])
ref = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
cmp_ = sys.argv[3] if len(sys.argv) > 3 else "ResNet50_TAMv2_after_layer4"
SEEDS = {f"seed_{x}" for x in sys.argv[4].split(",")} if len(sys.argv) > 4 else None


def seed_dirs(model):
    return [d for d in sorted((base / model).glob("seed_*")) if SEEDS is None or d.name in SEEDS]


def read_preds(f):
    d = pd.read_csv(f)
    if CLEAN_MASK is not None:
        if len(d) != len(CLEAN_MASK):
            sys.exit(f"{f} : {len(d)} lignes, masque de {len(CLEAN_MASK)} : manifeste incohérent.")
        d = d[CLEAN_MASK].reset_index(drop=True)
    return d


# ---- 1. Tableau global : moyenne ± std sur les seeds -------------------------------------------
rows = []
for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
    for seed_dir in seed_dirs(model_dir.name):
        f = seed_dir / "test_evaluation" / "test_predictions.csv"
        if not f.is_file():
            continue
        d = read_preds(f)
        rows.append({
            "model": model_dir.name, "seed": seed_dir.name,
            "acc": 100 * accuracy_score(d.true_label, d.predicted_label),
            "macro_f1": 100 * f1_score(d.true_label, d.predicted_label, average="macro", zero_division=0),
        })
runs = pd.DataFrame(rows)
if runs.empty:
    sys.exit("Aucun test_predictions.csv trouvé : vérifiez le dossier.")
agg = runs.groupby("model").agg(n_seeds=("seed", "count"), acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std")).round(2)
print("\n=== Test, moyenne ± écart-type sur les seeds (std = NaN si 1 seule seed) ===")
print(agg.to_string())
agg.to_csv(base / f"analysis_summary{SUFFIX}.csv")

# ---- 2. F1 par classe : référence vs comparé ----------------------------------------------------
def class_f1(model):
    out = []
    for seed_dir in seed_dirs(model):
        f = seed_dir / "test_evaluation" / "classification_report.csv"
        fp = seed_dir / "test_evaluation" / "test_predictions.csv"
        if CLEAN_MASK is not None and fp.is_file():
            d = read_preds(fp)
            labels = sorted(d.true_label.unique())
            out.append(pd.Series(f1_score(d.true_label, d.predicted_label, labels=labels, average=None, zero_division=0), index=labels))
        elif CLEAN_MASK is None and f.is_file():
            r = pd.read_csv(f, index_col=0)
            out.append(r["f1-score"])
    return pd.concat(out, axis=1).mean(axis=1) if out else None

a, b = class_f1(ref), class_f1(cmp_)
if a is not None and b is not None:
    classes = [c for c in a.index if c not in ("accuracy", "macro avg", "weighted avg")]
    tab = pd.DataFrame({ref: 100 * a[classes], cmp_: 100 * b[classes]})
    tab["diff"] = tab[cmp_] - tab[ref]
    print(f"\n=== F1 par classe (moyenne sur seeds), triées par différence ({cmp_} - {ref}) ===")
    print(tab.sort_values("diff").round(2).to_string())
    tab.round(2).to_csv(base / f"analysis_per_class{SUFFIX}.csv")

# ---- 3. McNemar exact par seed ------------------------------------------------------------------
print(f"\n=== McNemar exact : {ref} vs {cmp_} ===")
for seed_dir in seed_dirs(ref):
    s = seed_dir.name
    fa = seed_dir / "test_evaluation" / "test_predictions.csv"
    fb = base / cmp_ / s / "test_evaluation" / "test_predictions.csv"
    if not (fa.is_file() and fb.is_file()):
        continue
    da, db = read_preds(fa), read_preds(fb)
    if len(da) != len(db) or not (da.true_label.values == db.true_label.values).all():
        print(f"{s}: ordre ou taille des échantillons différents, comparaison impossible")
        continue
    ca, cb = da.correct.values.astype(bool), db.correct.values.astype(bool)
    n01 = int((~ca & cb).sum())   # seul le modèle comparé a raison
    n10 = int((ca & ~cb).sum())   # seule la référence a raison
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
    print(f"{s}: seul {cmp_} correct={n01}, seule {ref} correcte={n10}, p-value={p:.3f}")
print("\nLecture : p < 0,05 = différence significative pour cette seed ; sinon, modèles indiscernables.")