"""
pipelines/phase1_report.py
-----------------------------
Phase 1 — Rapport de collecte des datasets.

Produit un tableau récapitulatif des 5 datasets sources utilisés pour
construire WheatAI-Merged : nom, source, nombre d'images, tâche,
type d'annotation, date de collecte.

Réutilise les fonctions de scan de merge_datasets.py (une seule source
de vérité pour le comptage — pas de duplication de logique).

Usage :
    from pipelines.phase1_report import build_phase1_report
    build_phase1_report(config)
"""

import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

from pipelines.merge_datasets import (
    _collect_ds1, _collect_ds2, _collect_ds3, _collect_ds4, _collect_ds5_yolo
)


# Métadonnées fixes de chaque dataset (propriétés connues, pas déduites
# du scan — le scan ne donne que les comptages d'images).
DATASET_METADATA = {
    "DS1": {
        "name"      : "Wheat Plant Diseases",
        "source"    : "Kaggle — kushagra3204/wheat-plant-diseases",
        "task"      : "Classification",
        "annotation": "Dossiers par classe (folder-based labels)",
    },
    "DS2": {
        "name"      : "Disease Dataset of Wheat (Mendeley Bangladesh)",
        "source"    : "Mendeley Data — 5gc7hwydwg",
        "task"      : "Classification",
        "annotation": "Dossiers par classe (folder-based labels)",
    },
    "DS3": {
        "name"      : "Wheat-Rust-19 (Yellow Rust Hayit)",
        "source"    : "Kaggle — tolgahayit/yellowrust19",
        "task"      : "Classification (par sévérité)",
        "annotation": "Dossiers par code de sévérité (0/MR/MRMS/MS/R/S)",
    },
    "DS4": {
        "name"      : "Wheat Disease Dataset (Zenodo — Brown 2023)",
        "source"    : "Zenodo / Kaggle — wheat-disease-dataset-small",
        "task"      : "Classification",
        "annotation": "Dossiers par classe (folder-based labels)",
    },
    "DS5": {
        "name"      : "Wheat Rust Disease Computer Vision Dataset",
        "source"    : "Roboflow Universe (une ou plusieurs sources)",
        "task"      : "Object Detection",
        "annotation": "Bounding boxes YOLO (.txt), format YOLOv5-v11",
    },
}


def _get_folder_date(path: Path) -> str:
    """
    Estime une 'date de collecte' à partir de la date de modification
    du dossier racine du dataset. Ce n'est PAS la date de téléchargement
    réelle (non trackée automatiquement) — à corriger manuellement dans
    le CSV si tu connais la date exacte de téléchargement.
    """
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return "N/A"


def build_phase1_report(config: dict) -> pd.DataFrame:
    """
    Construit le tableau récapitulatif des datasets collectés (Phase 1).

    Args:
        config : configuration globale (config.yaml)

    Returns:
        DataFrame avec colonnes :
        Dataset Name, Source, Number of Images, Task, Annotation Type,
        Download Date (estimated)
    """
    print("\n" + "=" * 65)
    print("  PHASE 1 — Rapport de Collecte des Datasets")
    print("=" * 65)

    merge_cfg = config.get("merge", {})
    paths_cfg = merge_cfg.get("paths", {})
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n  Scan des datasets sources...\n")

    # DS1-DS4 : classification (réutilise les collecteurs existants)
    collectors = {
        "DS1": (_collect_ds1, "ds1"),
        "DS2": (_collect_ds2, "ds2"),
        "DS3": (_collect_ds3, "ds3"),
        "DS4": (_collect_ds4, "ds4"),
    }

    for ds_key, (collector_fn, path_key) in collectors.items():
        records = collector_fn(paths_cfg)
        meta = DATASET_METADATA[ds_key]
        folder_path = paths_cfg.get(path_key, "N/A")

        rows.append({
            "Dataset Name"    : meta["name"],
            "Source"          : meta["source"],
            "Number of Images": len(records),
            "Task"            : meta["task"],
            "Annotation Type" : meta["annotation"],
            "Download Date (estimated)": _get_folder_date(Path(folder_path)),
        })

    # DS5 : détection (une ou plusieurs sources Roboflow)
    ds5_results = _collect_ds5_yolo(paths_cfg)
    ds5_total = sum(len(r["images"]) for r in ds5_results)
    meta = DATASET_METADATA["DS5"]

    ds5_cfg = paths_cfg.get("ds5", "N/A")
    ds5_paths = ds5_cfg if isinstance(ds5_cfg, list) else [ds5_cfg]
    ds5_date = _get_folder_date(Path(ds5_paths[0])) if ds5_paths else "N/A"

    rows.append({
        "Dataset Name"    : meta["name"],
        "Source"          : f"{meta['source']} ({len(ds5_results)} source(s))",
        "Number of Images": ds5_total,
        "Task"            : meta["task"],
        "Annotation Type" : meta["annotation"],
        "Download Date (estimated)": ds5_date,
    })

    df = pd.DataFrame(rows)

    print("\n  " + "-" * 100)
    print(df.to_string(index=False))
    print("  " + "-" * 100)

    total_images = df["Number of Images"].sum()
    print(f"\n  Total (toutes sources, avant fusion/déduplication) : {total_images} images")

    # Sauvegarde
    csv_path = reports_dir / "collected_datasets.csv"
    json_path = reports_dir / "collected_datasets.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_images_all_sources": int(total_images),
            "datasets": rows,
            "note": ("'Download Date' est estimée depuis la date de modification "
                      "du dossier local, PAS la date de téléchargement réelle. "
                      "À corriger manuellement si connue avec précision."),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Rapport sauvegardé :")
    print(f"     {csv_path}")
    print(f"     {json_path}")
    print("\n  ✅ Phase 1 report terminé\n")

    return df