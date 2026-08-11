"""
main.py — Orchestrateur Principal
===================================
Thèse : Deep Learning, Computer Vision, and IoT for
        Real-Time Monitoring and Epidemic Forecasting in Agriculture
 
Auteur : Tarik Ismail
Lab    : MDSET — Hassan First University, Settat
 
Exécution dans Google Colab :
    !python main.py --phase 1
    !python main.py --phase 1 --kaggle_json /content/kaggle.json
 
Structure des phases :
    Phase 1 → Acquisition, Validation, Split, Preprocessing
    Phase 2 → Modèles Baseline (ResNet50, EfficientNetB0, YOLOv8)   [à venir]
    Phase 3 → Generative AI (GAN, cGAN, VAE)                         [à venir]
    Phase 4 → Unsupervised Learning (SOM, DBN, RBM)                  [à venir]
    Phase 5 → Multimodal Fusion (CNN + LSTM, Transformers)            [à venir]
"""
 
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
        raise FileNotFoundError(f"❌ Fichier config introuvable : {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"  ✅ Config chargée : {config_path}")
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
    print("  🌾 WHEAT RUST AI PIPELINE")
    print("  Deep Learning & Computer Vision for Agriculture")
    print("  MDSET Lab — Hassan First University, Settat")
    print("=" * 60)
    print(f"  ▶  Exécution : Phase {phase}")
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
    print("  ✅  PHASE 1 — RÉSUMÉ")
    print("=" * 60)
    print(f"  Dataset        : PlantVillage")
    print(f"  Cible          : Wheat Rust (Yellow / Brown / Stem + Healthy)")
    print(f"  Framework      : TensorFlow / Keras")
    print(f"  Images totales : {total}")
    print(f"  Classes        : {config['classes']['num_classes']}")
    print(f"  Image size     : {config['preprocessing']['img_size']}")
    print(f"  Normalisation  : [0, 1]")
    print(f"  Split          : {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    print(f"  Augmentation   : ✅ (8 transformations)")
    print()
    print(f"  📁 Fichiers générés :")
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
    print(f"  ➡️   PROCHAINE ÉTAPE : Phase 2 — Modèles Baseline")
    print(f"       (ResNet50 / EfficientNetB0 / YOLOv8)")
    print("=" * 60 + "\n")
 
 
# ─────────────────────────────────────────────────────────────
# Phases futures (stubs)
# ─────────────────────────────────────────────────────────────
 
def run_phase2(config: dict, models: list = None,
               skip_existing: bool = True, task: str = "both",
               data_source: str = "original") -> dict:
    """
    Phase 2 — Modèles Baseline (Classification + Détection).
 
    Recharge les artefacts déjà préparés (CSV + métadonnées) pour ne pas
    avoir à refaire le téléchargement/preprocessing, puis :
      - task='classification' : entraîne CNN custom / ResNet50 / EfficientNetB0
      - task='detection'      : entraîne YOLO — bbox=full_image si
                                 data_source='original' (pas d'annotations
                                 réelles), ou vraies bboxes Roboflow si
                                 data_source='merged'
      - task='both' (défaut)  : exécute les deux à la suite
 
    Args:
        config        : configuration globale
        models        : sous-ensemble de modèles de classification à
                        (re)traiter, ex: ["efficientnetb0"].
        skip_existing : si True, un modèle déjà entraîné (fichiers présents
                        sur disque) est rechargé au lieu d'être ré-entraîné.
        task          : "classification" | "detection" | "both"
        data_source   : "original" (Phase 1, 15 classes, data/processed/)
                        ou "merged" (5 datasets fusionnés, 17 classes,
                        data/merged/processed/). Les résultats sont
                        sauvegardés dans des dossiers de sortie distincts
                        pour ne pas écraser les résultats de l'autre source.
    """
    import pandas as pd
 
    if data_source == "merged":
        processed_dir = Path(config["merge"]["output"]) / "processed"
        train_csv = processed_dir / "merged_train.csv"
        val_csv   = processed_dir / "merged_val.csv"
        test_csv  = processed_dir / "merged_test.csv"
        meta_path = processed_dir / "merged_metadata.json"
        cls_output_dir = "outputs/phase2_classification_merged"
        det_output_dir = "outputs/phase2_detection_merged"
        merged_yolo_yaml = Path(config["merge"]["output"]) / "yolo" / "data.yaml"
    else:
        processed_dir = Path(config["paths"]["processed"])
        train_csv = processed_dir / "train.csv"
        val_csv   = processed_dir / "val.csv"
        test_csv  = processed_dir / "test.csv"
        meta_path = Path(config["paths"]["metadata_json"])
        cls_output_dir = config["phase2"]["paths"]["classification_outputs"]
        det_output_dir = config["phase2"]["paths"]["detection_outputs"]
        merged_yolo_yaml = None
 
    for p in (train_csv, val_csv, test_csv, meta_path):
        if not p.exists():
            hint = ("Lance d'abord la fusion : python main.py --phase merge"
                     if data_source == "merged"
                     else "Lance d'abord la Phase 1 : python main.py --phase 1")
            raise FileNotFoundError(f"❌ Fichier manquant : {p}\n   → {hint}")
 
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
 
    label2idx = metadata["label2idx"]
    idx2label = {int(k): v for k, v in metadata["idx2label"].items()}
    class_weights = {int(k): v for k, v in metadata["class_weights"].items()}
    config["classes"]["num_classes"] = metadata["num_classes"]
 
    # Redirige les dossiers de sortie sans modifier le fichier config.yaml
    config["phase2"]["paths"]["classification_outputs"] = cls_output_dir
    config["phase2"]["paths"]["detection_outputs"] = det_output_dir
 
    split_result = {
        "train_df"     : pd.read_csv(train_csv),
        "val_df"       : pd.read_csv(val_csv),
        "test_df"      : pd.read_csv(test_csv),
        "label2idx"    : label2idx,
        "idx2label"    : idx2label,
        "class_weights": class_weights,
    }
 
    print(f"  ✅ Dataset source : {data_source} — artefacts rechargés depuis {processed_dir}")
    print(f"     Train: {len(split_result['train_df'])} | "
          f"Val: {len(split_result['val_df'])} | "
          f"Test: {len(split_result['test_df'])} | "
          f"Classes: {config['classes']['num_classes']}")
    print(f"     Sorties classification : {cls_output_dir}")
    print(f"     Sorties détection      : {det_output_dir}")
 
    results = {}
 
    if task in ("classification", "both"):
        from pipelines.preprocessing import build_tf_datasets
        from pipelines.train_classification import run_classification_phase
 
        train_ds, val_ds, test_ds = build_tf_datasets(split_result, config)
        results["classification"] = run_classification_phase(
            split_result, train_ds, val_ds, test_ds, config,
            models=models, skip_existing=skip_existing
        )
 
    if task in ("detection", "both"):
        from pipelines.train_detection import run_detection_phase
 
        if data_source == "merged" and merged_yolo_yaml and merged_yolo_yaml.exists():
            print(f"\n  ℹ️  Détection avec VRAIES bboxes (Roboflow) : {merged_yolo_yaml}")
            results["detection"] = run_detection_phase(
                split_result, config, skip_existing=skip_existing,
                data_yaml_override=str(merged_yolo_yaml)
            )
        else:
            if data_source == "merged":
                print(f"\n  ⚠️  data.yaml fusionné introuvable ({merged_yolo_yaml}) "
                      f"— fallback bbox=full_image")
            results["detection"] = run_detection_phase(
                split_result, config, skip_existing=skip_existing
            )
 
    return results
 
 
def run_phase3(config: dict, classes: list = None) -> dict:
    """
    Phase 3 — Generative AI & Dataset Enrichment (cGAN + VAE).
 
    Recharge les artefacts de la Phase 1 (train.csv + métadonnées) pour
    identifier les classes minoritaires, puis entraîne un cGAN et un VAE
    conditionnel pour chacune, génère des échantillons synthétiques et
    les évalue (FID, SSIM).
 
    Args:
        config  : configuration globale
        classes : sous-ensemble explicite de labels à traiter, ex:
                  ["Stem_fly", "Black_Rust"]. None = auto-détection des
                  classes minoritaires via phase3.minority_selection.
    """
    import pandas as pd
    from pipelines.train_generative import run_generative_phase
 
    processed_dir = Path(config["paths"]["processed"])
    train_csv = processed_dir / "train.csv"
    meta_path = Path(config["paths"]["metadata_json"])
 
    for p in (train_csv, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"❌ Fichier manquant : {p}\n"
                f"   → Lance d'abord la Phase 1 : python main.py --phase 1"
            )
 
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
 
    idx2label = {int(k): v for k, v in metadata["idx2label"].items()}
    config["classes"]["num_classes"] = metadata["num_classes"]
 
    split_result = {
        "train_df" : pd.read_csv(train_csv),
        "idx2label": idx2label,
    }
 
    print(f"  ✅ Artefacts Phase 1 rechargés depuis {processed_dir}")
    print(f"     Train: {len(split_result['train_df'])} | "
          f"Classes: {config['classes']['num_classes']}")
 
    return run_generative_phase(split_result, config, classes=classes)
 
 
 
def run_merge(config: dict, copy_images: bool = False) -> dict:
    """
    Merge — Fusion des 5 datasets en un dataset unifié (WheatAI-Merged).
 
    Collecte, normalise les classes, déduplique, split et sauvegarde
    train/val/test CSV + metadata.json + sous-dataset YOLO (DS5 Roboflow).
 
    Args:
        config      : configuration globale
        copy_images : si True, copie physique des images (lourd).
                      Par défaut False = symlinks (recommandé sur Colab).
    """
    from pipelines.merge_datasets import build_merged_dataset
    return build_merged_dataset(config, copy_images=copy_images)
 
def run_phase4(config: dict) -> None:
    """Phase 4 — Unsupervised Learning (SOM, DBN, RBM) [à venir]"""
    print("  ⏳ Phase 4 non encore implémentée.")
    print("     → Implémentation prochaine : SOM, DBN, RBM")
 
 
def run_phase5(config: dict) -> None:
    """Phase 5 — Multimodal Fusion [à venir]"""
    print("  ⏳ Phase 5 non encore implémentée.")
    print("     → Implémentation prochaine : CNN+LSTM, Attention, Transformers")
 
 
def run_phase1_report(config: dict) -> "object":
    """
    Report — Résumé des datasets collectés (Phase 1).
 
    Produit un tableau récapitulatif (CSV + JSON) des 5 datasets sources :
    nom, source, nombre d'images, tâche, type d'annotation, date estimée.
    """
    from pipelines.phase1_report import build_phase1_report
    return build_phase1_report(config)
 
 
def run_dataset_validation_phase(config: dict, data_source: str = "merged",
                                  sample_size: int = 3000,
                                  remove_duplicates: bool = False) -> dict:
    """
    Validate — Validation complète du dataset (Phase 3 EDA approfondie).
 
    Génère automatiquement : distribution des classes, résolutions,
    formats, aspect ratios, modes couleur, images corrompues, doublons
    (perceptual hash), statistiques d'annotation YOLO/OBB, heatmap des
    bounding boxes, visualisations aléatoires, exemples d'annotations,
    et un rapport PDF final assemblant toutes ces figures.
 
    Args:
        config            : configuration globale
        data_source       : "merged" ou "original"
        sample_size       : nombre d'images échantillonnées pour les
                            opérations coûteuses (résolution, format,
                            couleur, doublons)
        remove_duplicates : si True, supprime physiquement les doublons
                            détectés (par défaut False — rapport seul)
    """
    from pipelines.dataset_validation import run_dataset_validation
    return run_dataset_validation(
        config, data_source=data_source, sample_size=sample_size,
        remove_duplicates=remove_duplicates
    )
 
 
def run_obb_phase(config: dict, data_source: str = "merged",
                   skip_generation: bool = False, skip_existing: bool = True) -> dict:
    """
    OBB — Génération automatique d'annotations OBB + entraînement YOLO-OBB
    (Contribution 2).
 
    Étapes :
        1. Génère des annotations OBB à partir des bboxes axis-aligned
           existantes (segmentation colorimétrique + rectangle orienté
           minimal), sauf si skip_generation=True (réutilise un dataset
           OBB déjà généré)
        2. Entraîne un modèle YOLO-OBB (yolo11n-obb.pt) sur ce dataset
        3. Évalue et compare implicitement avec le YOLO standard déjà
           entraîné (mêmes métriques : mAP50, mAP50-95, precision, recall)
 
    Args:
        config          : configuration globale
        data_source     : "merged" (dataset fusionné, source des vraies
                          bboxes Roboflow) — seule option pertinente ici
        skip_generation : si True, saute la génération OBB et utilise
                          directement un dataset déjà présent
        skip_existing   : si True, recharge un modèle YOLO-OBB déjà
                          entraîné au lieu de le ré-entraîner
    """
    from pipelines.obb_generation import build_obb_dataset
 
    if data_source == "merged":
        yolo_root = Path(config["merge"]["output"]) / "yolo"
        obb_root  = Path(config["merge"]["output"]) / "yolo_obb"
    else:
        yolo_root = Path(config["phase2"]["paths"]["yolo_dataset"])
        obb_root  = yolo_root.parent / "yolo_obb"
 
    if not yolo_root.exists():
        raise FileNotFoundError(
            f"❌ Dataset YOLO source introuvable : {yolo_root}\n"
            f"   → Lance d'abord la fusion : python main.py --phase merge"
        )
 
    obb_data_yaml = obb_root / "data.yaml"
 
    if skip_generation and obb_data_yaml.exists():
        print(f"  ⏭️  Génération OBB ignorée — dataset existant : {obb_data_yaml}")
    else:
        build_obb_dataset(str(yolo_root), str(obb_root))
 
    # Entraînement YOLO-OBB
    output_dir = Path(config["phase2"]["paths"]["detection_outputs"]).parent / "phase2_detection_obb"
    output_dir.mkdir(parents=True, exist_ok=True)
 
    metrics_path = output_dir / "metrics.json"
    best_weights_path = output_dir / "yolo_obb_train" / "weights" / "best.pt"
 
    if skip_existing and metrics_path.exists() and best_weights_path.exists():
        print(f"\n  ⏭️  Modèle YOLO-OBB déjà entraîné — rechargement")
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  mAP50: {metrics.get('mAP50', 'N/A')}")
        return {"metrics": metrics, "weights_path": str(best_weights_path)}
 
    from ultralytics import YOLO
 
    det_cfg = config["phase2"]["detection"]
    obb_model_name = det_cfg["model"].replace(".pt", "-obb.pt")
    if "-obb" not in obb_model_name:
        obb_model_name = "yolo11n-obb.pt"
 
    print(f"\n  Entraînement YOLO-OBB ({obb_model_name})...")
    model = YOLO(obb_model_name)
    model.train(
        data=str(obb_data_yaml),
        epochs=det_cfg["epochs"],
        imgsz=det_cfg["imgsz"],
        batch=det_cfg["batch"],
        project=str(output_dir),
        name="yolo_obb_train",
        exist_ok=True,
    )
 
    val_results = model.val(
        data=str(obb_data_yaml), split="test",
        project=str(output_dir), name="yolo_obb_eval", exist_ok=True,
    )
 
    metrics = {
        "mAP50"    : float(val_results.box.map50),
        "mAP50_95" : float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall"   : float(val_results.box.mr),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"\n  mAP50: {metrics['mAP50']:.4f}  mAP50-95: {metrics['mAP50_95']:.4f}")
    print(f"  ✅ YOLO-OBB entraîné — comparer avec outputs/phase2_detection*/metrics.json "
          f"(YOLO standard) pour l'analyse comparative de l'article")
 
    return {"metrics": metrics, "weights_path": str(best_weights_path)}
 
 
# ─────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────
 
def phase_type(value: str):
    """
    Convertit l'argument --phase : accepte soit un entier (1-5), soit un
    mot-clé ("merge", "report", "validate", "obb"). Évite toute ambiguïté
    avec argparse type=int qui rejetterait les mots-clés.
    """
    if value.isdigit():
        return int(value)
    return value.lower()


PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_merge,
    7: run_phase1_report,
    8: run_dataset_validation_phase,
    9: run_obb_phase,
}



def main():
    parser = argparse.ArgumentParser(
        description="Wheat Rust AI Pipeline — MDSET Lab",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--phase", type=phase_type, default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Phase à exécuter : 1=Preprocessing, 2=Baseline, 3=GAN, "
             "4=Unsupervised, 5=Fusion, 6=merge, 7=report, 8=validate, 9=obb"
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
    parser.add_argument(
        "--models", type=str, default=None,
        help="Phase 2 uniquement. Liste de modèles séparés par des virgules "
             "à (re)traiter, ex: --models efficientnetb0 "
             "ou --models resnet50,efficientnetb0. "
             "Par défaut : tous les modèles de config.yaml."
    )
    parser.add_argument(
        "--no_skip_existing", action="store_true",
        help="Phase 2 uniquement. Force le ré-entraînement même si un "
             "modèle a déjà été entraîné lors d'une session précédente "
             "(par défaut, les modèles déjà présents sur disque sont "
             "rechargés sans ré-entraînement)."
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Phase 3 uniquement. Liste de classes séparées par des virgules "
             "à traiter explicitement, ex: --classes Stem_fly,Black_Rust. "
             "Par défaut : auto-détection des classes minoritaires."
    )
    parser.add_argument(
        "--copy_images", action="store_true",
        help="Merge uniquement. Copie physique des images au lieu de symlinks."
    )
    parser.add_argument(
        "--task", type=str, default="both",
        choices=["classification", "detection", "both"],
        help="Phase 2 uniquement. 'classification' (CNN/ResNet/EfficientNet), "
             "'detection' (YOLO) ou 'both' (les deux, défaut)."
    )
    parser.add_argument(
        "--data_source", type=str, default="original",
        choices=["original", "merged"],
        help="Phase 2/validate. 'original' = Phase 1 seule (15 classes, "
             "data/processed/). 'merged' = 5 datasets fusionnés (17 classes, "
             "data/merged/processed/, vraies bboxes YOLO si disponibles). "
             "Résultats sauvegardés dans des dossiers séparés."
    )
    parser.add_argument(
        "--sample_size", type=int, default=3000,
        help="Phase validate uniquement. Nombre d'images échantillonnées "
             "pour les opérations coûteuses (résolution, format, couleur, "
             "doublons). La distribution de classes utilise toujours le "
             "dataset complet."
    )
    parser.add_argument(
        "--remove_duplicates", action="store_true",
        help="Phase validate uniquement. Supprime physiquement les "
             "doublons détectés (pHash). Par défaut : rapport seul, "
             "non destructif."
    )
    args = parser.parse_args()

    # Chargement de la config
    print_banner(args.phase)
    config = load_config(args.config)
    set_seeds(config["project"]["seed"])

    # Dispatch vers la phase sélectionnée
    runner = PHASE_RUNNERS.get(args.phase)
    if runner is None:
        print(f"❌ Phase {args.phase} inconnue.")
        sys.exit(1)

    if args.phase == 1:
        runner(config, kaggle_json=args.kaggle_json, skip_download=args.skip_download)
    elif args.phase == 2:
        models_list = (
            [m.strip() for m in args.models.split(",")] if args.models else None
        )
        runner(config, models=models_list, skip_existing=not args.no_skip_existing,
               task=args.task, data_source=args.data_source)
    elif args.phase == "merge":
        runner(config, copy_images=args.copy_images)
    elif args.phase == "report":
        runner(config)
    elif args.phase == "validate":
        runner(config, data_source=args.data_source, sample_size=args.sample_size,
               remove_duplicates=args.remove_duplicates)
    elif args.phase == 3:
        classes_list = (
            [c.strip() for c in args.classes.split(",")] if args.classes else None
        )
        runner(config, classes=classes_list)
    else:
        runner(config)


if __name__ == "__main__":
    main()