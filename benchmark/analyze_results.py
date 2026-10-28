"""Analyse des résultats du benchmark (à lancer sur Colab ou en local, sans GPU).

Usage :
    python analyze_results.py /content/drive/MyDrive/wheat_attention_v2 ResNet50 ResNet50_TAMv2_after_layer4

Arguments : dossier de sortie du benchmark, modèle de référence, modèle à comparer (noms de dossiers).
Lit, pour chaque seed, <dossier>/<modèle>/seed_N/test_evaluation/{test_predictions.csv, classification_report.csv}.
Produit : tableau moyenne ± écart-type de tous les modèles, F1 par classe (réf. vs comparé), McNemar par seed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

base = Path(sys.argv[1])
ref = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
cmp_ = sys.argv[3] if len(sys.argv) > 3 else "ResNet50_TAMv2_after_layer4"


def load_preds(model, seed_dir):
    return pd.read_csv(seed_dir / "test_evaluation" / "test_predictions.csv")


# ---- 1. Tableau global : moyenne ± std sur les seeds -------------------------------------------
rows = []
for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
    for seed_dir in sorted(model_dir.glob("seed_*")):
        f = seed_dir / "test_evaluation" / "test_predictions.csv"
        if not f.is_file():
            continue
        d = pd.read_csv(f)
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
agg.to_csv(base / "analysis_summary.csv")

# ---- 2. F1 par classe : référence vs comparé ----------------------------------------------------
def class_f1(model):
    out = []
    for seed_dir in sorted((base / model).glob("seed_*")):
        f = seed_dir / "test_evaluation" / "classification_report.csv"
        if f.is_file():
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
    tab.round(2).to_csv(base / "analysis_per_class.csv")

# ---- 3. McNemar exact par seed ------------------------------------------------------------------
print(f"\n=== McNemar exact : {ref} vs {cmp_} ===")
for seed_dir in sorted((base / ref).glob("seed_*")):
    s = seed_dir.name
    fa = seed_dir / "test_evaluation" / "test_predictions.csv"
    fb = base / cmp_ / s / "test_evaluation" / "test_predictions.csv"
    if not (fa.is_file() and fb.is_file()):
        continue
    da, db = pd.read_csv(fa), pd.read_csv(fb)
    if len(da) != len(db) or not (da.true_label.values == db.true_label.values).all():
        print(f"{s}: ordre ou taille des échantillons différents, comparaison impossible")
        continue
    ca, cb = da.correct.values.astype(bool), db.correct.values.astype(bool)
    n01 = int((~ca & cb).sum())   # seul le modèle comparé a raison
    n10 = int((ca & ~cb).sum())   # seule la référence a raison
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
    print(f"{s}: seul {cmp_} correct={n01}, seule {ref} correcte={n10}, p-value={p:.3f}")
print("\nLecture : p < 0,05 = différence significative pour cette seed ; sinon, modèles indiscernables.")
