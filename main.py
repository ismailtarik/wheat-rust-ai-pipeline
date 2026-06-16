import argparse
import json
import os
import sys
import yaml
import random
import numpy as np

from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Charge le fichier de configuration YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f" Fichier config introuvable : {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"  Config chargée : {config_path}")
    return config


def set_seeds(seed: int) -> None:
    """Fixe toutes les graines pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def print_banner(phase: int) -> None:
    print("\n" + "=" * 60)
    print("   WHEAT RUST AI PIPELINE")
    print("  Deep Learning & Computer Vision for Agriculture")
    print("  MDSET Lab — Hassan First University, Settat")
    print("=" * 60)
    print(f"  Exécution : Phase {phase}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# Phase 1 — Acquisition & Preprocessing
# ─────────────────────────────────────────────────────────────

def run_phase1(config: dict, kaggle_json: str = None,
               skip_download: bool = False) -> dict:
    """
    Exécute la Phase 1 complète :
        Étape 1 → data_collection  : Téléchargement PlantVillage (ou données locales)
        Étape 2 → validation       : Filtrage Wheat + EDA
        Étape 3 → split_dataset    : Split stratifié + exports
        Étape 4 → preprocessing    : Pipelines tf.data
    """
    from pipelines.data_collection import collect_data
    from pipelines.validation      import validate_dataset
    from pipelines.split_dataset   import split_dataset
    from pipelines.preprocessing   import build_tf_datasets

    # ── Étape 1 : Collecte ──
    collection_result = collect_data(
        config, kaggle_json_path=kaggle_json, skip_download=skip_download
    )

    # ── Étape 2 : Validation & EDA ──
    validation_result = validate_dataset(collection_result, config)

    # ── Étape 3 : Split ──
    split_result = split_dataset(validation_result, config)

    # ── Étape 4 : Pipelines tf.data ──
    train_ds, val_ds, test_ds = build_tf_datasets(split_result, config)

    # ── Résumé final ──
    _print_phase1_summary(split_result, config)

    return {
        "collection"  : collection_result,
        "validation"  : validation_result,
        "split"       : split_result,
        "datasets"    : (train_ds, val_ds, test_ds),
    }


def _print_phase1_summary(split_result: dict, config: dict) -> None:
    """Affiche le résumé de la Phase 1."""
    train_df = split_result["train_df"]
    val_df   = split_result["val_df"]
    test_df  = split_result["test_df"]
    total    = len(train_df) + len(val_df) + len(test_df)

    print("\n" + "=" * 60)
    print("   PHASE 1 — RÉSUMÉ")
    print("=" * 60)
    print(f"  Dataset        : PlantVillage")
    print(f"  Cible          : Wheat Rust (Yellow / Brown / Stem + Healthy)")
    print(f"  Framework      : TensorFlow / Keras")
    print(f"  Images totales : {total}")
    print(f"  Classes        : {config['classes']['num_classes']}")
    print(f"  Image size     : {config['preprocessing']['img_size']}")
    print(f"  Normalisation  : [0, 1]")
    print(f"  Split          : {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    print(f"  Augmentation   : (8 transformations)")
    print()
    print(f"   Fichiers générés :")
    print(f"     data/processed/train.csv")
    print(f"     data/processed/val.csv")
    print(f"     data/processed/test.csv")
    print(f"     data/processed/metadata.json")
    print(f"     data/reports/01_class_distribution.png")
    print(f"     data/reports/02_image_dimensions.png")
    print(f"     data/reports/03_sample_images.png")
    print(f"     data/reports/04_augmentation_preview.png")
    print(f"     data/reports/05_split_distribution.png")
    print()
    print(f"     PROCHAINE ÉTAPE : Phase 2 — Modèles Baseline")
    print(f"       (ResNet50 / EfficientNetB0 / YOLOv8)")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# Phases futures (stubs)
# ─────────────────────────────────────────────────────────────

def run_phase2(config: dict) -> None:
    """Phase 2 — Modèles Baseline (ResNet50, EfficientNetB0, YOLOv8) [à venir]"""
    print("   Phase 2 non encore implémentée.")
    print("     → Implémentation prochaine : ResNet50, EfficientNetB0, YOLOv8")


def run_phase3(config: dict) -> None:
    """Phase 3 — Generative AI (GAN, cGAN, VAE) [à venir]"""
    print("   Phase 3 non encore implémentée.")
    print("     → Implémentation prochaine : GAN, cGAN, VAE")


def run_phase4(config: dict) -> None:
    """Phase 4 — Unsupervised Learning (SOM, DBN, RBM) [à venir]"""
    print("   Phase 4 non encore implémentée.")
    print("     → Implémentation prochaine : SOM, DBN, RBM")


def run_phase5(config: dict) -> None:
    """Phase 5 — Multimodal Fusion [à venir]"""
    print("   Phase 5 non encore implémentée.")
    print("     → Implémentation prochaine : CNN+LSTM, Attention, Transformers")


# ─────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────

PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
}


def main():
    parser = argparse.ArgumentParser(
        description=" Wheat Rust AI Pipeline — MDSET Lab",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3, 4, 5],
        help="Phase à exécuter (1=Preprocessing, 2=Baseline, 3=GAN, 4=Unsupervised, 5=Fusion)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--kaggle_json", type=str, default=None,
        help="Chemin vers kaggle.json (Phase 1 uniquement)"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Ignore le téléchargement Kaggle et utilise les données déjà "
             "présentes dans 'paths.raw_data' (config.yaml)"
    )
    args = parser.parse_args()

    # Chargement de la config
    print_banner(args.phase)
    config = load_config(args.config)
    set_seeds(config["project"]["seed"])

    # Dispatch vers la phase sélectionnée
    runner = PHASE_RUNNERS.get(args.phase)
    if runner is None:
        print(f" Phase {args.phase} inconnue.")
        sys.exit(1)

    if args.phase == 1:
        runner(config, kaggle_json=args.kaggle_json, skip_download=args.skip_download)
    else:
        runner(config)


if __name__ == "__main__":
    main()