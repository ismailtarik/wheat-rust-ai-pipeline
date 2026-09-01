"""
evaluation/comparison_report.py
-----------------------------------
Génère les livrables comparatifs demandés pour l'étude E0 (ResNet50) vs
E1 (+SE) vs E2 (+CBAM) vs E3 (+Triplet Attention) :

  1. results comparison/per_class_f1_comparison.csv
     Table : class | ResNet50 | SE | CBAM | Triplet
     (valeurs = F1-score par classe, lues depuis les
     classification_report.json réellement générés par
     pipelines/train_classification.py pour chaque expérience)

  2. results comparison/model_complexity.csv
     Colonnes : Model, Parameters, Trainable_Parameters, FLOPs,
     Model_Size_MB, Inference_Time_ms
     - Parameters / Trainable_Parameters / Inference_Time : mesurés
       réellement (construction du modèle + timing de forward passes).
     - FLOPs : calculé si une librairie fiable est disponible, sinon "NA"
       (jamais inventé — conforme à la consigne du projet).

IMPORTANT (anti-fabrication) :
    Ce script ne lit et n'affiche QUE des valeurs obtenues en exécutant
    du code réel (construction de modèle, mesure de temps, ou lecture de
    fichiers JSON produits par un entraînement réel). Si un résultat
    n'est pas disponible sur disque, la cellule correspondante est "NA" —
    jamais une valeur inventée.

Usage :
    python evaluation/comparison_report.py --data_source merged
    python evaluation/comparison_report.py --data_source merged --skip_complexity
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Correspondance nom d'expérience -> nom de modèle dans pipelines/models.py
EXPERIMENTS = {
    "E0_resnet50": "resnet50",
    "E1_resnet50_se": "resnet50_se",
    "E2_resnet50_cbam": "resnet50_cbam",
    "E3_resnet50_triplet": "resnet50_triplet",
}

# En-têtes lisibles pour les tableaux (courts, publication-ready)
DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "resnet50_se": "SE",
    "resnet50_cbam": "CBAM",
    "resnet50_triplet": "Triplet",
}


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _classification_report_path(output_dir: Path, model_name: str) -> Path:
    """
    Chemin réel produit par pipelines/train_classification.py :
    <output_dir>/<model_name>/classification_report.json
    """
    return output_dir / model_name / "classification_report.json"


# ─────────────────────────────────────────────────────────────
# 1. Per-class F1 comparison
# ─────────────────────────────────────────────────────────────

def build_per_class_f1_comparison(output_dir: Path, out_csv: Path) -> pd.DataFrame:
    """
    Lit classification_report.json (produit par sklearn.classification_report
    avec output_dict=True dans pipelines/train_classification.py) pour
    chaque modèle disponible, et construit la table comparative par classe.
    """
    per_model_f1 = {}   # display_name -> {class_name: f1}
    class_names_seen = set()

    for model_name, display_name in DISPLAY_NAMES.items():
        report_path = _classification_report_path(output_dir, model_name)
        if not report_path.exists():
            print(f"  ⚠️  Pas de résultat pour '{model_name}' "
                  f"({report_path} introuvable) — colonne remplie en NA")
            per_model_f1[display_name] = None
            continue

        with open(report_path) as f:
            report = json.load(f)

        class_f1 = {
            cls: vals["f1-score"]
            for cls, vals in report.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        }
        per_model_f1[display_name] = class_f1
        class_names_seen.update(class_f1.keys())

    class_names_sorted = sorted(class_names_seen)
    rows = []
    for cls in class_names_sorted:
        row = {"class": cls}
        for display_name, class_f1 in per_model_f1.items():
            row[display_name] = (
                round(class_f1[cls], 4)
                if class_f1 is not None and cls in class_f1
                else "NA"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  ✅ {out_csv}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Model complexity (params réels, FLOPs si fiable, inference time réel)
# ─────────────────────────────────────────────────────────────

def _try_compute_flops(model, input_shape: tuple):
    """
    Tente un calcul de FLOPs via keras-flops si disponible. Retourne "NA"
    si la librairie est absente ou si le calcul échoue — conformément à
    la consigne : ne jamais inventer une valeur de FLOPs.
    """
    try:
        from keras_flops import get_flops
        flops = get_flops(model, batch_size=1)
        return int(flops)
    except Exception:
        return "NA"


def _measure_inference_time_ms(model, input_shape: tuple, n_warmup: int = 3,
                                n_runs: int = 20) -> float:
    """
    Mesure réelle du temps d'inférence moyen par image (CPU ou GPU,
    selon l'environnement d'exécution), sur des tenseurs aléatoires
    (le timing ne dépend pas du contenu des images).
    """
    import tensorflow as tf
    x = tf.random.uniform((1,) + input_shape)

    for _ in range(n_warmup):
        _ = model(x, training=False)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x, training=False)
    elapsed = time.perf_counter() - t0

    return round((elapsed / n_runs) * 1000.0, 2)


def build_model_complexity_table(config: dict, out_csv: Path,
                                  use_imagenet_weights: bool = True) -> pd.DataFrame:
    """
    Construit chaque modèle (E0..E3) et mesure sa complexité réelle.
    Nécessite TensorFlow installé ; nécessite un accès réseau pour les
    poids ImageNet si use_imagenet_weights=True (sinon, poids aléatoires
    — les comptes de paramètres et le temps d'inférence restent valides,
    seule l'exactitude des poids change, sans impact sur ces métriques).
    """
    from pipelines.models import build_model

    img_size = tuple(config["preprocessing"]["img_size"])
    input_shape = img_size + (3,)
    num_classes = config["classes"].get("num_classes")
    if not isinstance(num_classes, int):
        raise ValueError(
            "config['classes']['num_classes'] doit être un entier résolu "
            "(pas 'auto') pour construire les modèles ici — relancer "
            "après la Phase 1/merge qui le calcule et le fige."
        )

    rows = []
    for model_name, display_name in DISPLAY_NAMES.items():
        print(f"\n  🔧 Construction de {display_name} ({model_name})...")
        model = build_model(model_name, input_shape, num_classes, freeze_base=True)

        total_params = int(model.count_params())
        trainable_params = int(sum(
            np.prod(v.shape) for v in model.trainable_weights
        ))
        model_size_mb = round(total_params * 4 / (1024 ** 2), 2)  # float32
        flops = _try_compute_flops(model, input_shape)
        inference_ms = _measure_inference_time_ms(model, input_shape)

        rows.append({
            "Model": display_name,
            "Parameters": total_params,
            "Trainable_Parameters": trainable_params,
            "FLOPs": flops,
            "Model_Size_MB": model_size_mb,
            "Inference_Time_ms": inference_ms,
        })
        print(f"     params={total_params:,}  trainable={trainable_params:,}  "
              f"flops={flops}  size={model_size_mb} MB  "
              f"inference={inference_ms} ms/image")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n  ✅ {out_csv}")
    return df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rapports comparatifs E0/E1/E2/E3 (ResNet50 vs SE vs CBAM vs Triplet Attention)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_source", type=str, default="merged",
                         choices=["original", "merged"])
    parser.add_argument("--skip_complexity", action="store_true",
                         help="Ne pas (re)construire les modèles pour la table de complexité "
                              "(utile si les poids ImageNet ne sont pas téléchargeables).")
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.data_source == "merged":
        classification_output_dir = Path("outputs/phase2_classification_merged")
    else:
        classification_output_dir = Path(config["phase2"]["paths"]["classification_outputs"])

    comparison_dir = Path("results") / "comparison"

    print("=" * 60)
    print("  RAPPORT COMPARATIF — E0 (ResNet50) vs E1 (SE) vs E2 (CBAM) vs E3 (Triplet)")
    print("=" * 60)

    print("\n[1/2] Table comparative per-class F1...")
    build_per_class_f1_comparison(
        classification_output_dir, comparison_dir / "per_class_f1_comparison.csv"
    )

    if not args.skip_complexity:
        print("\n[2/2] Table de complexité des modèles...")
        build_model_complexity_table(config, comparison_dir / "model_complexity.csv")
    else:
        print("\n[2/2] Table de complexité ignorée (--skip_complexity)")

    print("\n✅ Rapport comparatif terminé.")


if __name__ == "__main__":
    main()