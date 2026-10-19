"""Construction du benchmark de classification V3 (audit, deduplication, pHash, integrite, v1 -> v2 -> v3).

Usage : python build_dataset.py --list | --stage 00 | --stage 00,01 | --stage all
"""
import csv
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
try:
    from IPython.display import display
except ImportError:  # hors notebook
    display = print

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour `import common`

from common import (PROJECT, PROJECT_ROOT)

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PROJECT_DIR = Path('/content/wheat-rust-ai-pipeline')


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def stage_00_acquire_datasets():
    """Acquisition des 3 sources du benchmark : wheat-plant-diseases (Kaggle -> data/raw/data), Wheat-disease-dataset-small (Kaggle -> ZenodoWheatSmall), MendeleyWheat (zip depuis Drive).
    
    Origine : cellule(s) 16, 20, 22 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : 3 commande(s) shell (!...) convertie(s) en subprocess.run.
    
    Chemins references : /content/drive ; /content/drive/MyDrive/Colab Notebooks/Wheat Disease.zip ; data/raw/MendeleyWheat
    """

    # ══════════ cellule 16 ══════════
    pass  # [notebook] !pip install kaggle  -> voir requirements_benchmark.txt

    subprocess.run('kaggle datasets download -d kushagra3204/wheat-plant-diseases', shell=True, check=False)
    subprocess.run('unzip wheat-plant-diseases.zip -d /content/wheat-rust-ai-pipeline/data/raw', shell=True, check=False)

    # ══════════ cellule 20 ══════════
    subprocess.run('kaggle datasets download -d yasserhessein/wheat-disease-dataset-small -p /content/wheat-rust-ai-pipeline/data/raw/ZenodoWheatSmall --unzip', shell=True, check=False)

    # ══════════ cellule 22 ══════════
    # Monter Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Copier et extraire le ZIP depuis Drive

    # Adapter ce chemin selon où tu as mis le ZIP dans ton Drive
    zip_path = '/content/drive/MyDrive/Colab Notebooks/Wheat Disease.zip'
    extract_path = '/content/wheat-rust-ai-pipeline/data/raw/MendeleyWheat'

    print("Extraction en cours...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)

    # Vérifier la structure extraite
    for root, dirs, files in os.walk(extract_path):
        if files:
            print(f"{root} → {len(files)} images")


def stage_01_prepare_mendeley():
    """Collecte/hachage, conflits d'etiquettes, deduplication et partitions du dataset Mendeley (Original Dataset). Role deduit des sections de la cellule.
    
    Origine : cellule(s) 55 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : A VALIDER : le choix de cette version/ce role repose sur une deduction (voir CELL_MAP.md).
    
    Chemins references : data/raw/ ; data/processed/
    """

    # ══════════ cellule 55 ══════════


    # =========================================================
    # CONFIGURATION
    # =========================================================

    SOURCE = Path(
        "/content/wheat-rust-ai-pipeline/data/raw/"
    "MendeleyWheat/Wheat Disease/Original Dataset"
    )

    OUTPUT = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/"
    "MendeleyWheat_Cleaned"
    )

    SEED = 42

    random.seed(SEED)

    # =========================================================
    # 1. COLLECTE ET HASHAGE
    # =========================================================

    if not SOURCE.exists():
        raise FileNotFoundError(f"Source introuvable : {SOURCE}")

    hash_groups = defaultdict(list)

    for class_dir in sorted(SOURCE.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in EXTENSIONS:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

                hash_groups[digest].append({
                    "path": image_path,
                    "class": class_name
                })

    print("Images analysées :", sum(len(v) for v in hash_groups.values()))
    print("Groupes SHA-256 :", len(hash_groups))

    # =========================================================
    # 2. DÉTECTION DES CONFLITS D'ÉTIQUETTES
    # =========================================================

    conflicts = []

    for digest, items in hash_groups.items():
        labels = {item["class"] for item in items}

        if len(labels) > 1:
            conflicts.append((digest, items))

    if conflicts:
        print("\nATTENTION : images identiques avec plusieurs labels.")
        for digest, items in conflicts:
            print("\nHash :", digest)
            for item in items:
                print(item["class"], item["path"])

        raise ValueError(
            "Conflits d'étiquettes détectés. "
        "Corrigez-les avant de poursuivre."
        )

    # =========================================================
    # 3. DÉDUPLICATION
    # =========================================================

    unique_by_class = defaultdict(list)

    for digest, items in hash_groups.items():
        # Les doublons ont tous la même classe.
        # On conserve le chemin lexicographiquement le plus petit.
        selected = sorted(items, key=lambda x: str(x["path"]))[0]

        unique_by_class[selected["class"]].append(selected["path"])

    print("\nEffectifs après déduplication :")

    for class_name in sorted(unique_by_class):
        print(f"{class_name}: {len(unique_by_class[class_name])}")

    print(
        "Total unique :",
        sum(len(v) for v in unique_by_class.values())
    )

    # =========================================================
    # 4. CRÉATION DES PARTITIONS
    # =========================================================

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} existe déjà. "
        "Renommez-le ou supprimez-le manuellement avant de relancer."
        )

    for split in ["Train", "Validation", "Test"]:
        (OUTPUT / split).mkdir(parents=True, exist_ok=True)

    manifest = []

    for class_name, images in sorted(unique_by_class.items()):

        images = sorted(images)
        random.Random(SEED).shuffle(images)

        n = len(images)

        # 70 / 20 / 10
        n_train = round(n * 0.70)
        n_val = round(n * 0.20)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        partitions = {
            "Train": train_images,
            "Validation": val_images,
            "Test": test_images
        }

        for split, selected_images in partitions.items():

            destination = OUTPUT / split / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for index, source_path in enumerate(selected_images):

                filename = f"{class_name}_{index:05d}{source_path.suffix.lower()}"
                destination_path = destination / filename

                shutil.copy2(source_path, destination_path)

                manifest.append({
                    "split": split,
                    "class": class_name,
                    "filename": filename,
                    "source": str(source_path),
                    "sha256": hashlib.sha256(
                        source_path.read_bytes()
                    ).hexdigest()
                })

        print(
            f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
        )

    # =========================================================
    # 5. SAUVEGARDE DU MANIFEST
    # =========================================================

    csv_path = OUTPUT / "manifest.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "class", "filename", "source", "sha256"
            ]
        )
        writer.writeheader()
        writer.writerows(manifest)

    # =========================================================
    # 6. RÉSUMÉ FINAL
    # =========================================================

    summary = Counter(item["split"] for item in manifest)

    print("\n" + "=" * 55)
    print("MendeleyWheat_Cleaned - RÉSULTAT")
    print("=" * 55)

    print("Total :", len(manifest))
    print("Train :", summary["Train"])
    print("Validation :", summary["Validation"])
    print("Test :", summary["Test"])
    print("Classes :", len(unique_by_class))
    print("Manifest :", csv_path)
    print("Dossier :", OUTPUT)


def stage_02_global_audit():
    """Audit global corrige : inventaire SHA-256 des 3 sources, conflits d'etiquettes, fuites intra-dataset, doublons inter-datasets -> data/reports/global_audit_corrected.
    
    Origine : cellule(s) 57 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/raw/data ; data/processed/MendeleyWheat_Cleaned ; data/raw/ZenodoWheatSmall/Wheat Disease Dataset ; data/reports/global_audit_corrected
    """

    # ══════════ cellule 57 ══════════


    # ============================================================
    # 1. PATHS
    # ============================================================


    SOURCES = {
        "data": PROJECT / "data/raw/data",
        "MendeleyWheat": PROJECT / "data/processed/MendeleyWheat_Cleaned",
        "ZenodoWheatSmall": (
            PROJECT / "data/raw/ZenodoWheatSmall/Wheat Disease Dataset"
        ),
    }

    OUTPUT = PROJECT / "data/reports/global_audit_corrected"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    KNOWN_PARTITIONS = {"Train", "Validation", "Test"}

    # ============================================================
    # 2. NORMALISATION DES CLASSES
    # ============================================================

    ALIASES = {
        "aphid": "Aphid",
        "black rust": "Black_Rust",
        "black_rust": "Black_Rust",
        "blast": "Wheat_Blast",
        "wheatblast": "Wheat_Blast",
        "wheat blast": "Wheat_Blast",
        "brown rust": "Brown_Rust",
        "brownrust": "Brown_Rust",
        "common root rot": "Common_Root_Rot",
        "common_root_rot": "Common_Root_Rot",
        "fusarium head blight": "Fusarium_Head_Blight",
        "fusarium_head_blight": "Fusarium_Head_Blight",
        "healthy": "Healthy",
        "healthyleaf": "Healthy",
        "healthy leaf": "Healthy",
        "leaf blight": "Leaf_Blight",
        "leafblight": "Leaf_Blight",
        "mildew": "Powdery_Mildew",
        "powdery_mildew": "Powdery_Mildew",
        "mite": "Mite",
        "septoria": "Septoria",
        "smut": "Smut",
        "stem fly": "Stem_Fly",
        "stem_fly": "Stem_Fly",
        "tan spot": "Tan_Spot",
        "tan_spot": "Tan_Spot",
        "yellow rust": "Yellow_Rust",
        "yellowrust": "Yellow_Rust",
        "yellow_rust": "Yellow_Rust",
        "blackpoint": "Black_Point",
        "black_point": "Black_Point",
        "fusariumfootrot": "Fusarium_Foot_Rot",
        "fusarium_foot_rot": "Fusarium_Foot_Rot",
    }

    def normalize_class(name):
        key = name.strip().lower()
        key = re.sub(r"[_\s]+", "_", key)

        # Suppression des suffixes liés aux partitions.
        for suffix in ["_test_valid", "_validation", "_valid", "_test"]:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                break

        return ALIASES.get(key, ALIASES.get(key.replace("_", " "), key))


    # ============================================================
    # 3. IDENTIFICATION CLASSE / PARTITION
    # ============================================================

    def get_metadata(path, root, dataset):

        relative = path.relative_to(root)
        parts = list(relative.parts)

        if dataset == "data":
            lower_parts = [p.lower() for p in parts]

            partition_index = None
            partition = "Unassigned"

            for i, part in enumerate(lower_parts):
                if part == "train":
                    partition_index = i
                    partition = "Train"
                    break
                elif part == "test":
                    partition_index = i
                    partition = "Test"
                    break
                elif part in ("valid", "validation"):
                    partition_index = i
                    partition = "Validation"
                    break

            if partition_index is None or partition_index + 1 >= len(parts):
                return "UNKNOWN", partition

            original_label = parts[partition_index + 1]

        elif dataset == "MendeleyWheat":
            lower_parts = [p.lower() for p in parts]

            partition = "Unassigned"
            if "train" in lower_parts:
                partition = "Train"
            elif "validation" in lower_parts:
                partition = "Validation"
            elif "test" in lower_parts:
                partition = "Test"

            original_label = path.parent.name

        else:
            partition = "Unassigned"
            original_label = path.parent.name

        return normalize_class(original_label), partition


    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


    # ============================================================
    # 4. INVENTAIRE
    # ============================================================

    records = []

    for dataset, root in SOURCES.items():

        print(f"\nAnalyse : {dataset}")
        print("Chemin :", root)

        if not root.exists():
            print("ATTENTION : source introuvable")
            continue

        count = 0

        for path in sorted(root.rglob("*")):

            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            label, partition = get_metadata(path, root, dataset)

            records.append({
                "dataset": dataset,
                "label": label,
                "partition": partition,
                "path": str(path),
                "filename": path.name,
                "sha256": sha256(path),
            })

            count += 1

        print("Images :", count)

    # ============================================================
    # 5. GROUPEMENT PAR HASH
    # ============================================================

    hash_groups = defaultdict(list)

    for record in records:
        hash_groups[record["sha256"]].append(record)

    duplicate_groups = {
        h: group for h, group in hash_groups.items()
        if len(group) > 1
    }

    label_conflicts = {}
    cross_dataset_duplicates = {}
    same_dataset_split_leaks = {}
    global_split_overlaps = {}

    for h, group in duplicate_groups.items():

        labels = {r["label"] for r in group}
        datasets = {r["dataset"] for r in group}

        known_splits = {
            r["partition"] for r in group
            if r["partition"] in KNOWN_PARTITIONS
        }

        # Même image, étiquettes canoniques différentes.
        if len(labels) > 1:
            label_conflicts[h] = group

        # Même image dans plusieurs datasets.
        if len(datasets) > 1:
            cross_dataset_duplicates[h] = group

        # Même dataset, plusieurs partitions connues.
        for dataset in datasets:
            dataset_splits = {
                r["partition"] for r in group
                if r["dataset"] == dataset
                and r["partition"] in KNOWN_PARTITIONS
            }

            if len(dataset_splits) > 1:
                same_dataset_split_leaks[h] = group
                break

        # Chevauchement de partitions connues, tous datasets confondus.
        if len(known_splits) > 1:
            global_split_overlaps[h] = group

    # ============================================================
    # 6. EXPORT CSV
    # ============================================================

    fields = [
        "dataset", "label", "partition",
        "filename", "path", "sha256"
    ]

    def export_records(filename, groups):

        output_file = OUTPUT / filename

        with open(output_file, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for h, group in groups.items():
                for r in group:
                    writer.writerow(r)

        return output_file


    with open(OUTPUT / "global_image_inventory.csv",
              "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


    export_records("duplicate_groups.csv", duplicate_groups)
    export_records("label_conflicts.csv", label_conflicts)
    export_records("cross_dataset_duplicates.csv", cross_dataset_duplicates)
    export_records("same_dataset_split_leaks.csv", same_dataset_split_leaks)
    export_records("global_split_overlaps.csv", global_split_overlaps)

    # ============================================================
    # 7. STATISTIQUES
    # ============================================================

    dataset_counts = Counter(r["dataset"] for r in records)
    class_counts = Counter(
        (r["dataset"], r["label"]) for r in records
    )
    partition_counts = Counter(
        (r["dataset"], r["partition"]) for r in records
    )

    lines = [
        "=" * 65,
        "CORRECTED GLOBAL DATASET AUDIT",
        "=" * 65,
        f"Total images: {len(records)}",
        f"Unique SHA-256 images: {len(hash_groups)}",
        f"Duplicate groups: {len(duplicate_groups)}",
        f"Duplicate occurrences: {len(records) - len(hash_groups)}",
        f"Cross-dataset duplicate groups: {len(cross_dataset_duplicates)}",
        f"Label conflict groups: {len(label_conflicts)}",
        f"Same-dataset split leakage groups: {len(same_dataset_split_leaks)}",
        f"Global split overlap groups: {len(global_split_overlaps)}",
        "",
        "IMAGES PER DATASET",
    ]

    for key, value in sorted(dataset_counts.items()):
        lines.append(f"{key}: {value}")

    lines.append("\nCLASSES PER DATASET")

    for (dataset, label), value in sorted(class_counts.items()):
        lines.append(f"{dataset} | {label}: {value}")

    lines.append("\nPARTITIONS PER DATASET")

    for (dataset, partition), value in sorted(partition_counts.items()):
        lines.append(f"{dataset} | {partition}: {value}")

    summary = "\n".join(lines)

    (OUTPUT / "global_audit_summary.txt").write_text(
        summary, encoding="utf-8"
    )

    print("\n" + summary)
    print("\nRapports sauvegardés dans :", OUTPUT)


def stage_04_deduplicate_v1():
    """Deduplication SHA-256 avec priorite de source + mise en quarantaine des conflits d'etiquettes -> data/processed/deduplication_v1.
    
    Origine : cellule(s) 69 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/reports/global_audit_corrected ; data/processed/deduplication_v1
    """

    # ══════════ cellule 69 ══════════


    # ============================================================
    # 1. CONFIGURATION
    # ============================================================

    ROOT = Path("/content/wheat-rust-ai-pipeline")

    REPORT_DIR = ROOT / "data/reports/global_audit_corrected"

    OUTPUT_DIR = (
        ROOT / "data/processed/deduplication_v1"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INVENTORY_PATH = REPORT_DIR / "global_image_inventory.csv"
    CONFLICT_PATH = REPORT_DIR / "label_conflicts.csv"

    # Priorité de provenance, déterministe
    SOURCE_PRIORITY = {
        "data": 1,
        "MendeleyWheat": 2,
        "ZenodoWheatSmall": 3
    }

    # ============================================================
    # 2. CHARGEMENT
    # ============================================================

    inventory = pd.read_csv(INVENTORY_PATH)
    conflicts = pd.read_csv(CONFLICT_PATH)

    assert len(inventory) == 16628
    assert inventory["sha256"].nunique() == 12857

    # Les SHA identifiés comme conflictuels dans le rapport
    conflict_hashes = set(conflicts["sha256"].dropna().unique())

    print("Inventaire :", len(inventory))
    print("SHA uniques :", inventory["sha256"].nunique())
    print("SHA conflictuels :", len(conflict_hashes))

    # ============================================================
    # 3. GROUPEMENT PAR SHA-256
    # ============================================================

    groups = defaultdict(list)

    for record in inventory.to_dict("records"):
        groups[record["sha256"]].append(record)

    # ============================================================
    # 4. CONSTRUCTION DU MANIFESTE
    # ============================================================

    manifest_rows = []
    summary_rows = []
    stats = Counter()

    for digest, group_records in groups.items():

        labels = sorted({
            str(r["label"])
            for r in group_records
            if pd.notna(r["label"]) and str(r["label"]).strip()
        })

        datasets = sorted({
            str(r["dataset"])
            for r in group_records
        })

        partitions = sorted({
            str(r["partition"])
            for r in group_records
            if pd.notna(r["partition"])
        })

        label_conflict = digest in conflict_hashes

        # Vérification supplémentaire : plusieurs labels
        if len(labels) > 1:
            label_conflict = True

        # Choix déterministe du représentant
        ordered_records = sorted(
            group_records,
            key=lambda r: (
                SOURCE_PRIORITY.get(r["dataset"], 999),
                r["path"]
            )
        )

        representative_record = ordered_records[0]

        if label_conflict:

            group_decision = "QUARANTINE"
            reason = "LABEL_CONFLICT"
            representative_path = ""

            stats["quarantine_groups"] += 1

            for record in group_records:

                manifest_rows.append({
                    **record,
                    "group_size": len(group_records),
                    "group_labels": "|".join(labels),
                    "group_datasets": "|".join(datasets),
                    "decision": "QUARANTINE",
                    "reason": reason,
                    "representative_path": "",
                    "keep": False
                })

        else:

            group_decision = "KEEP"
            representative_path = representative_record["path"]

            stats["consistent_groups"] += 1

            if len(group_records) == 1:
                reason = "UNIQUE_IMAGE"
            else:
                reason = "EXACT_DUPLICATE_SAME_LABEL"

            stats["duplicate_occurrences_removed"] += (
                len(group_records) - 1
            )

            for index, record in enumerate(ordered_records):

                if index == 0:
                    decision = "KEEP"
                    keep = True
                    stats["keep"] += 1
                else:
                    decision = "DUPLICATE"
                    keep = False
                    stats["duplicate"] += 1

                manifest_rows.append({
                    **record,
                    "group_size": len(ordered_records),
                    "group_labels": "|".join(labels),
                    "group_datasets": "|".join(datasets),
                    "decision": decision,
                    "reason": reason,
                    "representative_path": representative_path,
                    "keep": keep
                })

        summary_rows.append({
            "sha256": digest,
            "group_size": len(group_records),
            "labels": "|".join(labels),
            "datasets": "|".join(datasets),
            "partitions": "|".join(partitions),
            "decision": group_decision,
            "reason": reason,
            "representative_path": representative_path
        })

    # ============================================================
    # 5. DATAFRAMES
    # ============================================================

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)

    # ============================================================
    # 6. CONTROLES
    # ============================================================

    n_keep = int((manifest["decision"] == "KEEP").sum())
    n_duplicate = int((manifest["decision"] == "DUPLICATE").sum())
    n_quarantine = int((manifest["decision"] == "QUARANTINE").sum())

    assert len(manifest) == 16628
    assert len(summary) == 12857
    assert n_keep + n_duplicate + n_quarantine == 16628

    assert manifest["sha256"].nunique() == 12857

    assert summary["decision"].value_counts().to_dict() == {
        "KEEP": 12601,
        "QUARANTINE": 256
    }

    assert n_keep == 12601
    assert n_duplicate == 3445
    assert n_quarantine == 582

    assert manifest[
        manifest["decision"] == "QUARANTINE"
    ]["sha256"].nunique() == 256

    # ============================================================
    # 7. EXPORT
    # ============================================================

    manifest_path = OUTPUT_DIR / "global_dedup_manifest.csv"
    summary_path = OUTPUT_DIR / "dedup_group_summary.csv"
    quarantine_path = OUTPUT_DIR / "quarantine_images.csv"

    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest[
        manifest["decision"] == "QUARANTINE"
    ].to_csv(quarantine_path, index=False)

    # ============================================================
    # 8. RAPPORT
    # ============================================================

    print("\n" + "=" * 65)
    print("GLOBAL DEDUPLICATION MANIFEST - RECONSTRUCTED")
    print("=" * 65)

    print("Fichiers inventoriés :", len(manifest))
    print("Groupes SHA uniques :", len(summary))
    print("Groupes cohérents :", stats["consistent_groups"])
    print("Groupes en quarantaine :", stats["quarantine_groups"])

    print("\nDécisions par fichier :")
    print(manifest["decision"].value_counts())

    print("\nImages conservées par source :")
    print(
        manifest[manifest["keep"]]
        ["dataset"]
        .value_counts()
    )

    print("\nImages conservées par classe :")
    print(
        manifest[manifest["keep"]]
        ["label"]
        .value_counts()
        .sort_index()
    )

    print("\nFichiers générés :")
    print(manifest_path)
    print(summary_path)
    print(quarantine_path)

    print("\nTOUS LES CONTROLES SONT PASSES.")


def stage_05_build_classification_v1():
    """Reconstruction du benchmark 70/15/15 stratifie -> data/processed/classification_v1 (12 601 images).
    
    Origine : cellule(s) 73 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/deduplication_v1 ; data/processed/classification_v1
    """

    # ══════════ cellule 73 ══════════

    # ============================================================
    # RECONSTRUCTION DU BENCHMARK DE CLASSIFICATION
    # 70% TRAIN / 15% VALIDATION / 15% TEST
    # ============================================================


    # ------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------


    DEDUP_DIR = PROJECT_DIR / "data/processed/deduplication_v1"
    DEDUP_MANIFEST = DEDUP_DIR / "global_dedup_manifest.csv"

    OUTPUT_DIR = PROJECT_DIR / "data/processed/classification_v1"

    RANDOM_STATE = 42

    assert DEDUP_MANIFEST.exists(), "Manifeste global introuvable."

    # Ne jamais écraser un benchmark existant
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Le dossier {OUTPUT_DIR} contient déjà des fichiers. "
        "Arrêt pour éviter tout écrasement."
        )

    # ------------------------------------------------------------
    # 2. CHARGEMENT DES IMAGES KEEP
    # ------------------------------------------------------------

    global_df = pd.read_csv(DEDUP_MANIFEST)

    keep_df = global_df[
        global_df["decision"].astype(str).str.upper() == "KEEP"
    ].copy()

    keep_df = keep_df.reset_index(drop=True)

    keep_df["image_path"] = keep_df["path"]
    keep_df["canonical_class"] = keep_df["label"]

    print("=" * 65)
    print("SOURCE DATASET")
    print("=" * 65)

    print("Images KEEP :", len(keep_df))
    print("Classes :", keep_df["canonical_class"].nunique())

    assert len(keep_df) == 12601
    assert keep_df["sha256"].nunique() == len(keep_df)
    assert keep_df["canonical_class"].nunique() == 17

    # ------------------------------------------------------------
    # 3. VERIFICATION DES SOURCES
    # ------------------------------------------------------------

    missing_sources = [
        p for p in keep_df["image_path"]
        if not Path(p).is_file()
    ]

    assert not missing_sources, (
        f"{len(missing_sources)} fichiers sources sont absents. "
    f"Exemple : {missing_sources[:5]}"
    )

    print("Tous les fichiers sources existent.")

    # ------------------------------------------------------------
    # 4. CREATION DES PARTITIONS STRATIFIEES
    # ------------------------------------------------------------

    indices = keep_df.index
    labels = keep_df["canonical_class"]

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    temporary_labels = keep_df.loc[
        temporary_indices, "canonical_class"
    ]

    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_labels
    )

    train_df = keep_df.loc[train_indices].copy()
    val_df = keep_df.loc[val_indices].copy()
    test_df = keep_df.loc[test_indices].copy()

    train_df["new_split"] = "train"
    val_df["new_split"] = "val"
    test_df["new_split"] = "test"

    print("\n" + "=" * 65)
    print("PARTITIONING")
    print("=" * 65)

    print("Train :", len(train_df))
    print("Validation :", len(val_df))
    print("Test :", len(test_df))

    assert len(train_df) + len(val_df) + len(test_df) == 12601

    # ------------------------------------------------------------
    # 5. CREATION DES DOSSIERS ET COPIE DES IMAGES
    # ------------------------------------------------------------

    split_dataframes = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in split_dataframes.items():

        print(f"\nCopie de la partition : {split_name}")

        for idx, row in split_df.iterrows():

            source = Path(row["image_path"])

            extension = source.suffix.lower() or ".jpg"

            destination = (
                OUTPUT_DIR
                / split_name
                / row["canonical_class"]
                / (row["sha256"] + extension)
            )

            destination.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source, destination)

            split_df.at[idx, "destination_path"] = str(destination)

        print("Images copiées :", len(split_df))

    # ------------------------------------------------------------
    # 6. CREATION DU MANIFESTE FINAL
    # ------------------------------------------------------------

    final_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    final_manifest_path = OUTPUT_DIR / "classification_manifest.csv"

    final_df.to_csv(final_manifest_path, index=False)

    print("\nManifeste créé :", final_manifest_path)

    # ------------------------------------------------------------
    # 7. VERIFICATION DES COPIES ET DES HASHES
    # ------------------------------------------------------------

    def calculate_sha256(filepath):

        h = hashlib.sha256()

        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        return h.hexdigest()


    verification_errors = []

    for _, row in final_df.iterrows():

        destination = Path(row["destination_path"])

        if not destination.is_file():
            verification_errors.append({
                "path": str(destination),
                "error": "FILE_NOT_FOUND"
            })
            continue

        actual_hash = calculate_sha256(destination)

        if actual_hash != row["sha256"]:
            verification_errors.append({
                "path": str(destination),
                "error": "HASH_MISMATCH"
            })

    # ------------------------------------------------------------
    # 8. VERIFICATION DES FUITES ENTRE PARTITIONS
    # ------------------------------------------------------------

    hash_sets = {
        name: set(partition["sha256"])
        for name, partition in split_dataframes.items()
    }

    train_val_overlap = hash_sets["train"] & hash_sets["val"]
    train_test_overlap = hash_sets["train"] & hash_sets["test"]
    val_test_overlap = hash_sets["val"] & hash_sets["test"]

    leakage_count = (
        len(train_val_overlap)
        + len(train_test_overlap)
        + len(val_test_overlap)
    )

    # ------------------------------------------------------------
    # 9. BILAN FINAL
    # ------------------------------------------------------------

    physical_counts = {
        name: sum(
            1 for p in (OUTPUT_DIR / name).rglob("*")
            if p.is_file()
        )
        for name in ["train", "val", "test"]
    }

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK VERIFICATION")
    print("=" * 65)

    print("Train :", physical_counts["train"])
    print("Validation :", physical_counts["val"])
    print("Test :", physical_counts["test"])

    print("Total physique :", sum(physical_counts.values()))
    print("Total manifeste :", len(final_df))

    print("Erreurs fichiers/hashes :", len(verification_errors))
    print("Fuites SHA entre partitions :", leakage_count)

    if verification_errors:

        pd.DataFrame(verification_errors).to_csv(
            OUTPUT_DIR / "verification_errors.csv",
            index=False
        )

        raise RuntimeError(
            "La vérification physique a détecté des erreurs."
        )

    assert sum(physical_counts.values()) == len(final_df)
    assert len(final_df) == 12601
    assert leakage_count == 0

    print("\nBENCHMARK CREATED AND VERIFIED")
    print("Dossier :", OUTPUT_DIR)
    print("Manifeste :", final_manifest_path)

    print("\nRépartition par classe et partition :")
    display(
        final_df.groupby(
            ["canonical_class", "new_split"]
        ).size().unstack(fill_value=0)
    )


def stage_06_phash_near_duplicate_audit():
    """Audit des quasi-doublons par pHash (distance de Hamming, BK-tree) -> data/reports/near_duplicate_audit.
    
    Origine : cellule(s) 76 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v1 ; data/reports/near_duplicate_audit
    """

    # ══════════ cellule 76 ══════════

    # ============================================================
    # NEAR-DUPLICATE AUDIT — PHASH
    # WheatAI-Merged Classification Benchmark
    # ============================================================

    pass  # [notebook] !pip -q install ImageHash tqdm  -> voir requirements_benchmark.txt


    # ============================================================
    # 1. CONFIGURATION
    # ============================================================


    BENCHMARK_DIR = (
        PROJECT_ROOT / "data/processed/classification_v1"
    )

    MANIFEST_PATH = BENCHMARK_DIR / "classification_manifest.csv"

    REPORT_DIR = (
        PROJECT_ROOT / "data/reports/near_duplicate_audit"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Distance maximale entre deux pHash
    PHASH_THRESHOLD = 6

    # ============================================================
    # 2. CHARGEMENT ET VALIDATION
    # ============================================================

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "sha256",
        "canonical_class",
        "new_split",
        "destination_path"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    df = df.drop_duplicates(subset="sha256").reset_index(drop=True)

    assert len(df) == 12601, f"Nombre inattendu : {len(df)}"

    missing_files = [
        p for p in df["destination_path"]
        if not Path(p).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} images introuvables. "
        f"Exemple : {missing_files[:5]}"
        )

    print("=" * 70)
    print("PHASH NEAR-DUPLICATE AUDIT")
    print("=" * 70)

    print("Images :", len(df))
    print("Classes :", df["canonical_class"].nunique())
    print("Partitions :", df["new_split"].value_counts().to_dict())
    print("Seuil pHash :", PHASH_THRESHOLD)

    # ============================================================
    # 3. CALCUL DES EMPREINTES PHASH
    # ============================================================

    def calculate_phash(image_path):
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return int(str(imagehash.phash(img)), 16)

    phash_records = []
    errors = []

    start_time = time.time()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Calcul des pHash"
    ):
        try:
            value = calculate_phash(row["destination_path"])

            phash_records.append({
                "sha256": row["sha256"],
                "phash_int": value,
                "phash_hex": format(value, "016x"),
                "image_path": row["destination_path"],
                "canonical_class": row["canonical_class"],
                "split": row["new_split"]
            })

        except Exception as e:
            errors.append({
                "path": row["destination_path"],
                "error": str(e)
            })

    phash_df = pd.DataFrame(phash_records)

    phash_path = REPORT_DIR / "phash_inventory.csv"
    phash_df.drop(columns=["phash_int"]).to_csv(
        phash_path, index=False
    )

    errors_df = pd.DataFrame(errors)
    errors_path = REPORT_DIR / "phash_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    print("\nImages traitées :", len(phash_df))
    print("Erreurs :", len(errors))
    print("Temps de calcul :",
          round(time.time() - start_time, 2), "secondes")

    if len(phash_df) == 0:
        raise RuntimeError("Aucune empreinte pHash calculée.")

    # ============================================================
    # 4. BK-TREE POUR RECHERCHER LES PHASH PROCHES
    # ============================================================

    def hamming_distance(a, b):
        return (a ^ b).bit_count()


    class BKTree:

        def __init__(self):
            self.root = None

        def add(self, value):

            if self.root is None:
                self.root = {
                    "value": value,
                    "children": {}
                }
                return

            node = self.root

            while True:
                distance = hamming_distance(
                    value, node["value"]
                )

                if distance == 0:
                    return

                if distance not in node["children"]:
                    node["children"][distance] = {
                        "value": value,
                        "children": {}
                    }
                    return

                node = node["children"][distance]

        def search(self, value, radius):

            if self.root is None:
                return []

            results = []
            stack = [self.root]

            while stack:

                node = stack.pop()

                distance = hamming_distance(
                    value, node["value"]
                )

                if distance <= radius:
                    results.append((node["value"], distance))

                lower = distance - radius
                upper = distance + radius

                for edge, child in node["children"].items():
                    if lower <= edge <= upper:
                        stack.append(child)

            return results


    # ============================================================
    # 5. DÉTECTION DES PAIRES SIMILAIRES
    # ============================================================

    tree = BKTree()

    # Associe chaque pHash aux images qui le possèdent
    hash_to_records = defaultdict(list)

    pairs = []

    for record in tqdm(
        phash_records,
        desc="Recherche des quasi-doublons"
    ):

        current_hash = record["phash_int"]

        # Chercher les empreintes proches déjà rencontrées
        candidates = tree.search(
            current_hash,
            PHASH_THRESHOLD
        )

        for candidate_hash, distance in candidates:

            previous_records = hash_to_records[candidate_hash]

            for previous in previous_records:

                same_class = (
                    previous["canonical_class"]
                    == record["canonical_class"]
                )

                same_split = (
                    previous["split"] == record["split"]
                )

                pairs.append({
                    "image_1": previous["image_path"],
                    "image_2": record["image_path"],
                    "sha256_1": previous["sha256"],
                    "sha256_2": record["sha256"],
                    "class_1": previous["canonical_class"],
                    "class_2": record["canonical_class"],
                    "split_1": previous["split"],
                    "split_2": record["split"],
                    "phash_1": previous["phash_hex"],
                    "phash_2": record["phash_hex"],
                    "hamming_distance": distance,
                    "same_class": same_class,
                    "same_split": same_split,
                    "cross_split": not same_split
                })

        # Ajouter le hash à l'index
        if current_hash not in hash_to_records:
            tree.add(current_hash)

        hash_to_records[current_hash].append(record)


    # ============================================================
    # 6. EXPORT DES RÉSULTATS
    # ============================================================

    pairs_df = pd.DataFrame(pairs)

    pairs_path = REPORT_DIR / "near_duplicate_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)

    # Résumé par distance
    if len(pairs_df) > 0:

        distance_summary = (
            pairs_df["hamming_distance"]
            .value_counts()
            .sort_index()
            .rename_axis("hamming_distance")
            .reset_index(name="number_of_pairs")
        )

        distance_summary.to_csv(
            REPORT_DIR / "distance_distribution.csv",
            index=False
        )

        # Résumé par relation
        summary = pd.DataFrame([
            {
                "metric": "Total similar pairs",
                "value": len(pairs_df)
            },
            {
                "metric": "Same class pairs",
                "value": int(pairs_df["same_class"].sum())
            },
            {
                "metric": "Different class pairs",
                "value": int((~pairs_df["same_class"]).sum())
            },
            {
                "metric": "Cross partition pairs",
                "value": int(pairs_df["cross_split"].sum())
            },
            {
                "metric": "Same partition pairs",
                "value": int(pairs_df["same_split"].sum())
            }
        ])

    else:

        distance_summary = pd.DataFrame()
        summary = pd.DataFrame([
            {"metric": "Total similar pairs", "value": 0}
        ])

    summary.to_csv(
        REPORT_DIR / "near_duplicate_summary.csv",
        index=False
    )

    # ============================================================
    # 7. AFFICHAGE DU BILAN
    # ============================================================

    print("\n" + "=" * 70)
    print("NEAR-DUPLICATE AUDIT RESULTS")
    print("=" * 70)

    print("Images analysées :", len(phash_df))
    print("Images non traitées :", len(errors))
    print("Seuil Hamming :", PHASH_THRESHOLD)
    print("Paires similaires :", len(pairs_df))

    if len(pairs_df) > 0:

        print("\nRésumé :")
        print(summary.to_string(index=False))

        print("\nDistribution des distances :")
        print(distance_summary.to_string(index=False))

        print("\nPaires entre partitions :")
        display(
            pairs_df[pairs_df["cross_split"]]
            .sort_values("hamming_distance")
            .head(20)
        )

        print("\nPaires entre classes différentes :")
        display(
            pairs_df[~pairs_df["same_class"]]
            .sort_values("hamming_distance")
            .head(20)
        )

    else:
        print("Aucune paire détectée au seuil choisi.")

    print("\nFichiers générés :")
    print("Inventaire :", phash_path)
    print("Paires :", pairs_path)
    print("Résumé :", REPORT_DIR / "near_duplicate_summary.csv")
    print("Erreurs :", errors_path)

    print("\nAUDIT TERMINÉ.")


def stage_07b_phash_pair_review_sheets():
    """Planches de revue des paires (second generateur).
    
    Origine : cellule(s) 78 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : A VALIDER : le choix de cette version/ce role repose sur une deduction (voir CELL_MAP.md).
    
    Chemins references : data/reports/near_duplicate_audit
    """

    # ══════════ cellule 78 ══════════


    # ============================================================
    # CONFIGURATION
    # ============================================================

    BASE_DIR = Path(
        "/content/wheat-rust-ai-pipeline/data/reports/near_duplicate_audit"
    )

    PAIRS_FILE = BASE_DIR / "near_duplicate_pairs.csv"

    OUTPUT_DIR = (
        BASE_DIR / "visual_review" / "different_class_pairs"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PAIRS_PER_SHEET = 12
    THUMB_W = 200
    THUMB_H = 150

    # ============================================================
    # CHARGEMENT ET FILTRAGE
    # ============================================================

    df = pd.read_csv(PAIRS_FILE)

    filtered = df[
        (df["cross_split"] == True) &
        (df["same_class"] == False)
    ].copy()

    filtered = filtered.sort_values(
        by=["hamming_distance", "class_1", "class_2"]
    ).reset_index(drop=True)

    print("Nombre de paires :", len(filtered))

    # Sauvegarder les paires sélectionnées
    filtered.to_csv(
        OUTPUT_DIR / "different_class_pairs.csv",
        index=False
    )

    # ============================================================
    # GENERATION DES PLANCHES
    # ============================================================

    def create_sheet(batch, sheet_number):

        pair_height = THUMB_H + 95
        sheet_width = 2 * THUMB_W + 80
        sheet_height = len(batch) * pair_height + 60

        sheet = Image.new(
            "RGB",
            (sheet_width, sheet_height),
            "white"
        )

        draw = ImageDraw.Draw(sheet)
        y = 20

        for _, row in batch.iterrows():

            # Noms réels des colonnes du CSV
            path1 = str(row["image_1"])
            path2 = str(row["image_2"])

            try:
                img1 = Image.open(path1).convert("RGB")
                img2 = Image.open(path2).convert("RGB")

                img1.thumbnail((THUMB_W, THUMB_H))
                img2.thumbnail((THUMB_W, THUMB_H))

                x1 = 20
                x2 = THUMB_W + 60

                sheet.paste(img1, (x1, y))
                sheet.paste(img2, (x2, y))

                draw.text(
                    (x1, y + THUMB_H + 5),
                    f"Class: {row['class_1']}",
                    fill="black"
                )

                draw.text(
                    (x2, y + THUMB_H + 5),
                    f"Class: {row['class_2']}",
                    fill="black"
                )

                draw.text(
                    (20, y + THUMB_H + 25),
                    f"Distance: {row['hamming_distance']} | "
                f"Split 1: {row['split_1']} | "
                f"Split 2: {row['split_2']}",
                    fill="blue"
                )

                y += pair_height

            except Exception as e:
                print(
                    f"Erreur lors du chargement des images : "
                f"{path1} / {path2} — {e}"
                )

        output_path = OUTPUT_DIR / (
            f"different_class_sheet_{sheet_number:03d}.jpg"
        )

        sheet.save(output_path, quality=90)

        print("Planche créée :", output_path)


    # ============================================================
    # DECOUPAGE EN LOTS
    # ============================================================

    if len(filtered) == 0:

        print("Aucune paire de classes différentes.")

    else:

        total_sheets = math.ceil(
            len(filtered) / PAIRS_PER_SHEET
        )

        for i in range(total_sheets):

            start = i * PAIRS_PER_SHEET
            end = start + PAIRS_PER_SHEET

            batch = filtered.iloc[start:end]

            create_sheet(batch, i + 1)

        print("\n===================================")
        print("INSPECTION VISUELLE PREPAREE")
        print("===================================")
        print("Paires :", len(filtered))
        print("Planches :", total_sheets)
        print("Dossier :", OUTPUT_DIR)


def stage_07c_classify_phash_pairs():
    """Classification des paires pHash (meme classe / classes differentes, distance 0 ...).
    
    Origine : cellule(s) 81 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/reports/near_duplicate_audit
    """

    # ══════════ cellule 81 ══════════


    # ============================================================
    # CONFIGURATION
    # ============================================================

    BASE_DIR = Path(
        "/content/wheat-rust-ai-pipeline/data/reports/near_duplicate_audit"
    )

    INPUT_FILE = BASE_DIR / "near_duplicate_pairs.csv"

    OUTPUT_DIR = (
        BASE_DIR / "visual_review" / "different_class_pairs"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # CHARGEMENT DES 75 PAIRES
    # ============================================================

    df = pd.read_csv(INPUT_FILE)

    pairs = df[
        (df["cross_split"] == True) &
        (df["same_class"] == False)
    ].copy()

    # ============================================================
    # COMPARAISON SHA-256
    # ============================================================

    pairs["same_sha256"] = (
        pairs["sha256_1"].astype(str) ==
        pairs["sha256_2"].astype(str)
    )

    def classify_pair(row):

        if row["same_sha256"]:
            return "EXACT_BINARY_DUPLICATE_LABEL_CONFLICT"

        elif row["hamming_distance"] == 0:
            return "VISUAL_DUPLICATE_CANDIDATE_DISTANCE_0"

        else:
            return "NEAR_DUPLICATE_CANDIDATE"

    pairs["audit_status"] = pairs.apply(
        classify_pair,
        axis=1
    )

    # ============================================================
    # RAPPORT DETAILLE
    # ============================================================

    columns = [
        "image_1",
        "image_2",
        "sha256_1",
        "sha256_2",
        "class_1",
        "class_2",
        "split_1",
        "split_2",
        "hamming_distance",
        "same_sha256",
        "audit_status"
    ]

    report = pairs[columns].copy()

    report_path = OUTPUT_DIR / "cross_class_conflict_audit.csv"

    report.to_csv(report_path, index=False)

    # ============================================================
    # STATISTIQUES
    # ============================================================

    print("=" * 65)
    print("AUDIT DES PAIRES CROSS-SPLIT DE CLASSES DIFFERENTES")
    print("=" * 65)

    print("Nombre total de paires :", len(report))

    print("\nRésultats SHA-256 :")
    print(report["same_sha256"].value_counts())

    print("\nRépartition par statut :")
    print(report["audit_status"].value_counts())

    print("\nAssociations de classes :")
    print(
        report.groupby(
            ["class_1", "class_2", "audit_status"]
        ).size().sort_values(ascending=False)
    )

    print("\nRapport sauvegardé :", report_path)

    # ============================================================
    # AFFICHAGE DES DOUBLONS BINAIRES EVENTUELS
    # ============================================================

    exact_duplicates = report[
        report["same_sha256"] == True
    ]

    print("\nDoublons binaires avec conflit de classe :")
    display(exact_duplicates)


def stage_07d_init_manual_review_register():
    """Initialisation du registre de revue manuelle (manual_review_register.csv) a partir de
    cross_class_conflict_audit.csv (produit par l'etape 07c). Le registre est ensuite exige par l'etape 08.
    
    Origine : cellule(s) 82 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie.
    
    NOTE : cette cellule etait initialement classee a tort comme "diagnostic" ; elle produit un fichier
    requis par l'etape 08 (corrige apres execution reelle de l'etape 04/05 et analyse des dependances).
    """



    BASE_DIR = Path(
        "/content/wheat-rust-ai-pipeline/data/reports/near_duplicate_audit"
    )

    INPUT_FILE = (
        BASE_DIR / "visual_review" /
        "different_class_pairs" / "cross_class_conflict_audit.csv"
    )

    OUTPUT_FILE = (
        BASE_DIR / "visual_review" /
        "different_class_pairs" / "manual_review_register.csv"
    )

    df = pd.read_csv(INPUT_FILE)

    # Initialisation du registre de validation
    df["visual_decision"] = "NOT_REVIEWED"

    df["annotation_decision"] = "PENDING"

    df["review_comment"] = ""

    df["final_action"] = "NO_ACTION_YET"

    # Export
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("REGISTRE DE VALIDATION MANUELLE")
    print("=" * 60)

    print("Nombre de paires :", len(df))
    print("Paires pHash = 0 :", (df["hamming_distance"] == 0).sum())
    print("Paires pHash > 0 :", (df["hamming_distance"] > 0).sum())

    print("\nColonnes de validation :")
    print("visual_decision")
    print("annotation_decision")
    print("review_comment")
    print("final_action")

    print("\nFichier créé :", OUTPUT_FILE)


def stage_08_build_classification_v2():
    """Nettoyage des paires pHash de distance 0 inter-classes -> classification_v2 (12 518 images).
    
    Origine : cellule(s) 85 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v1 ; data/reports/near_duplicate_audit ; data/processed/classification_v2 ; data/reports/classification_v2
    """

    # ══════════ cellule 85 ══════════

    # ============================================================
    # CLEANING AUTOMATIQUE DES PAIRES pHASH DISTANCE 0
    # CREATION DE classification_v2
    # ============================================================



    # ------------------------------------------------------------
    # 1. CHEMINS
    # ------------------------------------------------------------


    CLASSIFICATION_DIR = (
        PROJECT_DIR / "data/processed/classification_v1"
    )

    AUDIT_DIR = (
        PROJECT_DIR / "data/reports/near_duplicate_audit"
    )

    PAIR_FILE = (
        AUDIT_DIR
        / "visual_review/different_class_pairs/different_class_pairs.csv"
    )

    REVIEW_FILE = (
        AUDIT_DIR
        / "visual_review/different_class_pairs/manual_review_register.csv"
    )

    MANIFEST_FILE = (
        CLASSIFICATION_DIR / "classification_manifest.csv"
    )

    OUTPUT_DIR = (
        PROJECT_DIR / "data/processed/classification_v2"
    )

    REPORT_DIR = (
        PROJECT_DIR / "data/reports/classification_v2"
    )

    for p in [PAIR_FILE, REVIEW_FILE, MANIFEST_FILE]:
        assert p.exists(), f"Fichier introuvable : {p}"

    # Ne jamais écraser une version existante
    assert not OUTPUT_DIR.exists(), (
        f"{OUTPUT_DIR} existe déjà. "
    "Choisir un nouveau dossier de sortie."
    )

    assert not REPORT_DIR.exists(), (
        f"{REPORT_DIR} existe déjà. "
    "Choisir un nouveau dossier de rapport."
    )

    # ------------------------------------------------------------
    # 2. CHARGEMENT
    # ------------------------------------------------------------

    manifest = pd.read_csv(MANIFEST_FILE)
    pairs = pd.read_csv(PAIR_FILE)
    reviews = pd.read_csv(REVIEW_FILE)

    print("Images initiales :", len(manifest))
    print("Paires inter-classes :", len(pairs))

    # ------------------------------------------------------------
    # 3. IDENTIFIER LES PAIRES DISTANCE 0
    #    ENTRE CLASSES DIFFERENTES
    # ------------------------------------------------------------

    cross_class_zero = pairs[
        (pairs["hamming_distance"] == 0)
        & (pairs["class_1"] != pairs["class_2"])
    ].copy()

    print(
        "\nPaires inter-classes avec distance 0 :",
        len(cross_class_zero)
    )

    # ------------------------------------------------------------
    # 4. EXTRAIRE LES HASHES DES IMAGES A QUARANTAINE
    # ------------------------------------------------------------

    quarantine_hashes = set(
        cross_class_zero["sha256_1"].dropna().astype(str)
    ) | set(
        cross_class_zero["sha256_2"].dropna().astype(str)
    )

    print(
        "Images distinctes candidates à la quarantaine :",
        len(quarantine_hashes)
    )

    # ------------------------------------------------------------
    # 5. SEPARER LES IMAGES A CONSERVER
    # ------------------------------------------------------------

    manifest["sha256"] = manifest["sha256"].astype(str)

    quarantine_df = manifest[
        manifest["sha256"].isin(quarantine_hashes)
    ].copy()

    clean_df = manifest[
        ~manifest["sha256"].isin(quarantine_hashes)
    ].copy()

    print("\nImages mises en quarantaine :", len(quarantine_df))
    print("Images conservées :", len(clean_df))

    # Vérification : aucune image retirée ne doit rester
    assert not set(clean_df["sha256"]) & quarantine_hashes

    # ------------------------------------------------------------
    # 6. VERIFIER LES PARTITIONS
    # ------------------------------------------------------------

    print("\nRépartition initiale après nettoyage :")
    print(clean_df["new_split"].value_counts())

    print("\nRépartition par classe :")
    display(
        clean_df.groupby(
            ["canonical_class", "new_split"]
        ).size().unstack(fill_value=0)
    )

    # ------------------------------------------------------------
    # 7. PREPARER LES DOSSIERS
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 8. COPIER LES IMAGES CONSERVEES
    # ------------------------------------------------------------

    clean_df = clean_df.reset_index(drop=True)

    new_destinations = []

    for _, row in clean_df.iterrows():

        source = Path(row["destination_path"])

        assert source.is_file(), f"Source absente : {source}"

        split = row["new_split"]
        class_name = row["canonical_class"]

        destination = (
            OUTPUT_DIR
            / split
            / class_name
            / source.name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(source, destination)

        new_destinations.append(str(destination))

    clean_df["destination_path_v2"] = new_destinations

    # ------------------------------------------------------------
    # 9. SAUVEGARDER LES MANIFESTES
    # ------------------------------------------------------------

    clean_df.to_csv(
        OUTPUT_DIR / "classification_manifest.csv",
        index=False
    )

    quarantine_df.to_csv(
        REPORT_DIR / "quarantined_images.csv",
        index=False
    )

    cross_class_zero.to_csv(
        REPORT_DIR / "cross_class_zero_distance_pairs.csv",
        index=False
    )

    # ------------------------------------------------------------
    # 10. VERIFICATION PHYSIQUE
    # ------------------------------------------------------------



    errors = []

    for _, row in clean_df.iterrows():

        path = Path(row["destination_path_v2"])

        if not path.is_file():
            errors.append((str(path), "FILE_NOT_FOUND"))
            continue

        if sha256_file(path) != row["sha256"]:
            errors.append((str(path), "HASH_MISMATCH"))

    assert not errors, f"Erreurs détectées : {errors[:5]}"

    # ------------------------------------------------------------
    # 11. VERIFICATION DES FUITES SHA
    # ------------------------------------------------------------

    split_hashes = {
        split: set(
            clean_df.loc[
                clean_df["new_split"] == split,
                "sha256"
            ]
        )
        for split in ["train", "val", "test"]
    }

    overlap_count = (
        len(split_hashes["train"] & split_hashes["val"])
        + len(split_hashes["train"] & split_hashes["test"])
        + len(split_hashes["val"] & split_hashes["test"])
    )

    assert overlap_count == 0

    # ------------------------------------------------------------
    # 12. BILAN
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("CLASSIFICATION V2 — VERIFICATION")
    print("=" * 65)

    print("Images initiales :", len(manifest))
    print("Images en quarantaine :", len(quarantine_df))
    print("Images conservées :", len(clean_df))

    print("\nPartitions :")
    print(clean_df["new_split"].value_counts())

    print("\nErreurs physiques :", len(errors))
    print("Fuites SHA :", overlap_count)

    print("\nDossier :", OUTPUT_DIR)
    print("Rapports :", REPORT_DIR)

    print("\nCLASSIFICATION V2 CREATED AND VERIFIED")


def stage_09a_integrity_split_audit():
    """Audit d'integrite sur v2 : rapport SHA-256, fuites entre partitions, paires inter-splits.
    
    Origine : cellule(s) 86 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v2 ; data/reports/near_duplicate_audit/near_duplicate_pairs.csv
    """

    # ══════════ cellule 86 ══════════


    # ============================================================
    # 1. CHEMINS
    # ============================================================

    BASE_DIR = Path("/content/wheat-rust-ai-pipeline")

    V2_DIR = BASE_DIR / "data/processed/classification_v2"
    MANIFEST_PATH = V2_DIR / "classification_manifest.csv"

    PHASH_PATH = (
        BASE_DIR
        / "data/reports/near_duplicate_audit/near_duplicate_pairs.csv"
    )

    REPORT_DIR = V2_DIR / "integrity_audit"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    assert MANIFEST_PATH.exists(), f"Manifeste introuvable : {MANIFEST_PATH}"
    assert PHASH_PATH.exists(), f"Rapport pHash introuvable : {PHASH_PATH}"

    # ============================================================
    # 2. CHARGEMENT DES DONNÉES
    # ============================================================

    manifest = pd.read_csv(MANIFEST_PATH)
    phash = pd.read_csv(PHASH_PATH)

    print("=" * 65)
    print("AUDIT D'INTÉGRITÉ — CLASSIFICATION_V2")
    print("=" * 65)

    print("\nColonnes du manifeste :")
    print(manifest.columns.tolist())

    print("\nColonnes du rapport pHash :")
    print(phash.columns.tolist())

    # Détection des colonnes
    def find_column(df, candidates, description):
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(
            f"Colonne introuvable pour {description}. "
        f"Colonnes disponibles : {df.columns.tolist()}"
        )

    hash_col = find_column(
        manifest,
        ["sha256", "SHA256", "sha"],
        "SHA-256 du manifeste"
    )

    split_col = find_column(
        manifest,
        ["new_split", "split", "partition"],
        "partition du manifeste"
    )

    class_col = find_column(
        manifest,
        ["canonical_class", "label", "class"],
        "classe du manifeste"
    )

    hash1_col = find_column(
        phash,
        ["sha256_1", "SHA256_1"],
        "SHA-256 image 1"
    )

    hash2_col = find_column(
        phash,
        ["sha256_2", "SHA256_2"],
        "SHA-256 image 2"
    )

    distance_col = find_column(
        phash,
        ["hamming_distance", "distance", "phash_distance"],
        "distance pHash"
    )

    # Normalisation des valeurs
    manifest[hash_col] = manifest[hash_col].astype(str).str.strip()
    manifest[split_col] = manifest[split_col].astype(str).str.strip()

    phash[hash1_col] = phash[hash1_col].astype(str).str.strip()
    phash[hash2_col] = phash[hash2_col].astype(str).str.strip()

    # ============================================================
    # 3. AUDIT SHA-256 : DOUBLONS EXACTS ENTRE LES PARTITIONS
    # ============================================================

    print("\n" + "=" * 65)
    print("ÉTAPE 1 — AUDIT SHA-256")
    print("=" * 65)

    # Une empreinte présente dans plusieurs partitions
    hash_split_counts = (
        manifest.groupby(hash_col)[split_col]
        .nunique()
    )

    leaking_hashes = hash_split_counts[hash_split_counts > 1].index

    sha_leakage = manifest[
        manifest[hash_col].isin(leaking_hashes)
    ].sort_values([hash_col, split_col])

    print(f"Images dans le manifeste : {len(manifest)}")
    print(f"Empreintes SHA-256 uniques : {manifest[hash_col].nunique()}")
    print(f"Empreintes présentes dans plusieurs partitions : {len(leaking_hashes)}")

    if len(leaking_hashes) == 0:
        print("RÉSULTAT : aucune fuite de doublons exacts détectée.")
    else:
        print("ATTENTION : des doublons exacts sont répartis entre plusieurs partitions.")

    sha_report = REPORT_DIR / "sha256_cross_split_leakage.csv"
    sha_leakage.to_csv(sha_report, index=False)

    # ============================================================
    # 4. AUDIT pHash : SIMILARITÉS ENTRE LES PARTITIONS
    # ============================================================

    print("\n" + "=" * 65)
    print("ÉTAPE 2 — AUDIT pHash")
    print("=" * 65)

    # Associer chaque empreinte SHA à sa partition et à sa classe
    # Les éventuels SHA dupliqués sont signalés par l'audit précédent.
    hash_info = (
        manifest
        .drop_duplicates(subset=[hash_col])
        .set_index(hash_col)[[split_col, class_col]]
    )

    # Ne conserver que les paires dont les deux images existent
    # encore dans classification_v2
    phash_current = phash[
        phash[hash1_col].isin(hash_info.index)
        & phash[hash2_col].isin(hash_info.index)
    ].copy()

    # Récupérer les partitions et classes actuelles
    phash_current["current_split_1"] = (
        phash_current[hash1_col].map(hash_info[split_col])
    )
    phash_current["current_split_2"] = (
        phash_current[hash2_col].map(hash_info[split_col])
    )

    phash_current["current_class_1"] = (
        phash_current[hash1_col].map(hash_info[class_col])
    )
    phash_current["current_class_2"] = (
        phash_current[hash2_col].map(hash_info[class_col])
    )

    # Paires situées dans des partitions différentes
    cross_split = phash_current[
        phash_current["current_split_1"]
        != phash_current["current_split_2"]
    ].copy()

    cross_split[distance_col] = pd.to_numeric(
        cross_split[distance_col], errors="coerce"
    )

    # Distances 0 à 6
    cross_split_0_6 = cross_split[
        cross_split[distance_col].between(0, 6)
    ].copy()

    # Distances strictement comprises entre 1 et 6
    cross_split_1_6 = cross_split[
        cross_split[distance_col].between(1, 6)
    ].copy()

    # Paires de classes différentes
    cross_class = cross_split_0_6[
        cross_split_0_6["current_class_1"]
        != cross_split_0_6["current_class_2"]
    ].copy()

    # Sauvegarde des résultats
    cross_split_0_6.to_csv(
        REPORT_DIR / "phash_cross_split_distance_0_6.csv",
        index=False
    )

    cross_split_1_6.to_csv(
        REPORT_DIR / "phash_cross_split_distance_1_6.csv",
        index=False
    )

    cross_class.to_csv(
        REPORT_DIR / "phash_cross_split_different_classes.csv",
        index=False
    )

    print(f"Paires pHash initiales : {len(phash)}")
    print(f"Paires encore présentes dans classification_v2 : {len(phash_current)}")
    print(f"Paires entre partitions différentes : {len(cross_split)}")
    print(f"Paires inter-partitions avec distance 0 à 6 : {len(cross_split_0_6)}")
    print(f"Paires inter-partitions avec distance 1 à 6 : {len(cross_split_1_6)}")
    print(f"Paires inter-partitions de classes différentes (distance 0 à 6) : {len(cross_class)}")

    # ============================================================
    # 5. RÉSUMÉ FINAL
    # ============================================================

    print("\n" + "=" * 65)
    print("RÉSUMÉ DE L'AUDIT")
    print("=" * 65)

    if len(leaking_hashes) == 0:
        print("SHA-256 : PASS — aucune fuite exacte entre partitions.")
    else:
        print("SHA-256 : WARNING — fuite exacte détectée.")

    if len(cross_split_1_6) == 0:
        print("pHash (distance 1-6) : aucune paire inter-partitions détectée.")
    else:
        print(
            f"pHash (distance 1-6) : {len(cross_split_1_6)} "
        "paire(s) à examiner."
        )

    print("\nRapports enregistrés dans :")
    print(REPORT_DIR)

    print("\nFichiers générés :")
    for file in sorted(REPORT_DIR.glob("*.csv")):
        print(f" - {file.name}")

    print("\nAudit terminé. Aucune image n'a été modifiée.")


def stage_09c_cross_class_visual_validation():
    """Chargement des paires inter-classes et validation visuelle manuelle.
    
    Origine : cellule(s) 88 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v2
    """

    # ══════════ cellule 88 ══════════


    # ============================================================
    # 1. CHEMINS
    # ============================================================

    BASE_DIR = Path("/content/wheat-rust-ai-pipeline")

    V2_DIR = BASE_DIR / "data/processed/classification_v2"
    AUDIT_DIR = V2_DIR / "integrity_audit"

    PAIR_PATH = AUDIT_DIR / "phash_cross_split_different_classes.csv"

    OUTPUT_PATH = (
        AUDIT_DIR / "cross_class_manual_validation_register.csv"
    )

    # ============================================================
    # 2. CHARGEMENT DES 21 PAIRES
    # ============================================================

    df = pd.read_csv(PAIR_PATH)

    # Déduplication indépendante de l'ordre des images
    df["pair_key"] = df.apply(
        lambda r: tuple(sorted([
            str(r["sha256_1"]),
            str(r["sha256_2"])
        ])),
        axis=1
    )

    df = df.drop_duplicates("pair_key").copy()
    df = df.sort_values(
        "hamming_distance",
        ascending=True
    ).reset_index(drop=True)

    # ============================================================
    # 3. OBSERVATIONS VISUELLES
    # ============================================================

    observations = {
        1: ("SIMILAR_AMBIGUOUS",
            "Deux vues très similaires d'un insecte sur une feuille."),
        2: ("SAME_OR_NEAR_IDENTICAL",
            "Même aspect général de l'épi de blé."),
        3: ("SAME_OR_NEAR_IDENTICAL",
            "Composition très similaire, photographie potentiellement identique."),
        4: ("SAME_OR_NEAR_IDENTICAL",
            "Composition très similaire à la paire 3."),
        5: ("SAME_OR_NEAR_IDENTICAL",
            "Triptyque visuellement presque identique."),
        6: ("SAME_OR_NEAR_IDENTICAL",
            "Même feuille et filigrane similaire."),
        7: ("SAME_OR_NEAR_IDENTICAL",
            "Même composition en trois panneaux."),
        8: ("SIMILAR_AMBIGUOUS",
            "Épis très similaires ; une image comporte du texte ajouté."),
        9: ("SAME_OR_NEAR_IDENTICAL",
            "Composition similaire aux paires 2, 13 et 16."),
        10: ("SAME_OR_NEAR_IDENTICAL",
             "Photographies de feuilles apparemment identiques."),
        11: ("SAME_OR_NEAR_IDENTICAL",
             "Même montage de feuilles présentant des symptômes."),
        12: ("SAME_OR_NEAR_IDENTICAL",
             "Composition très proche de la paire 11."),
        13: ("SAME_OR_NEAR_IDENTICAL",
             "Composition similaire aux paires 2, 9 et 16."),
        14: ("SAME_OR_NEAR_IDENTICAL",
             "Même montage de feuilles."),
        15: ("SIMILAR_AMBIGUOUS",
             "Même scène générale avec de nombreux petits insectes."),
        16: ("SAME_OR_NEAR_IDENTICAL",
             "Composition similaire aux paires 2, 9 et 13."),
        17: ("SAME_OR_NEAR_IDENTICAL",
             "Même photographie avec filigrane."),
        18: ("SIMILAR_AMBIGUOUS",
             "Montage de feuilles avec symptômes visuellement similaires."),
        19: ("SIMILAR_AMBIGUOUS",
             "Gros plan très similaire d'une feuille avec des pustules."),
        20: ("NOT_SAME_IMAGE",
             "Un épi de blé et un acarien isolé : contenus visuellement différents."),
        21: ("SAME_OR_NEAR_IDENTICAL",
             "Photographie très similaire à la paire 10.")
    }

    # Les numéros correspondent aux numéros affichés sur les planches.
    df["visual_pair_id"] = range(1, len(df) + 1)

    # Les observations ne sont associées que si l'ordre du CSV
    # correspond bien aux identifiants affichés dans les planches.
    if len(df) != 21:
        raise ValueError(
            f"21 paires attendues, {len(df)} trouvées. "
        "Vérifier le rapport avant de créer le registre."
        )

    df["visual_decision"] = df["visual_pair_id"].map(
        lambda i: observations[i][0]
    )

    df["visual_comment"] = df["visual_pair_id"].map(
        lambda i: observations[i][1]
    )

    # ============================================================
    # 4. CHAMPS DE VALIDATION
    # ============================================================

    df["annotation_decision"] = "PENDING"
    df["reviewer"] = ""
    df["review_date"] = ""
    df["final_action"] = "NO_ACTION_YET"
    df["review_comment"] = ""

    # ============================================================
    # 5. EXPORT
    # ============================================================

    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 65)
    print("REGISTRE DE VALIDATION CRÉÉ")
    print("=" * 65)

    print("Nombre de paires :", len(df))
    print("\nRépartition des décisions visuelles :")
    print(df["visual_decision"].value_counts())

    print("\nRépartition des décisions d'annotation :")
    print(df["annotation_decision"].value_counts())

    print("\nFichier enregistré :")
    print(OUTPUT_PATH)

    print("\nColonnes disponibles :")
    print(df.columns.tolist())


def stage_09d_cross_class_connected_groups():
    """Groupes connexes inter-classes (union-find).
    
    Origine : cellule(s) 89 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v2
    """

    # ══════════ cellule 89 ══════════


    # ============================================================
    # 1. CHEMINS
    # ============================================================

    BASE_DIR = Path("/content/wheat-rust-ai-pipeline")
    V2_DIR = BASE_DIR / "data/processed/classification_v2"
    AUDIT_DIR = V2_DIR / "integrity_audit"

    SOURCE_PAIRS = AUDIT_DIR / "phash_cross_split_different_classes.csv"
    REGISTER_PATH = AUDIT_DIR / "cross_class_manual_validation_register.csv"

    # ============================================================
    # 2. OBSERVATIONS VISUELLES PAR NUMÉRO ORIGINAL
    # ============================================================

    observations = {
        1: ("SIMILAR_AMBIGUOUS",
            "Deux vues très similaires d'un insecte sur une feuille."),
        2: ("SAME_OR_NEAR_IDENTICAL",
            "Même aspect général de l'épi de blé."),
        3: ("SAME_OR_NEAR_IDENTICAL",
            "Composition très similaire, photographie potentiellement identique."),
        4: ("SAME_OR_NEAR_IDENTICAL",
            "Composition très similaire à la paire 3."),
        5: ("SAME_OR_NEAR_IDENTICAL",
            "Triptyque visuellement presque identique."),
        6: ("SAME_OR_NEAR_IDENTICAL",
            "Même feuille et filigrane similaire."),
        7: ("SAME_OR_NEAR_IDENTICAL",
            "Même composition en trois panneaux."),
        8: ("SIMILAR_AMBIGUOUS",
            "Épis très similaires ; une image comporte du texte ajouté."),
        9: ("SAME_OR_NEAR_IDENTICAL",
            "Composition similaire aux paires 2, 13 et 16."),
        10: ("SAME_OR_NEAR_IDENTICAL",
             "Photographies de feuilles apparemment identiques."),
        11: ("SAME_OR_NEAR_IDENTICAL",
             "Même montage de feuilles présentant des symptômes."),
        12: ("SAME_OR_NEAR_IDENTICAL",
             "Composition très proche de la paire 11."),
        13: ("SAME_OR_NEAR_IDENTICAL",
             "Composition similaire aux paires 2, 9 et 16."),
        14: ("SAME_OR_NEAR_IDENTICAL",
             "Même montage de feuilles."),
        15: ("SIMILAR_AMBIGUOUS",
             "Même scène générale avec de nombreux petits insectes."),
        16: ("SAME_OR_NEAR_IDENTICAL",
             "Composition similaire aux paires 2, 9 et 13."),
        17: ("SAME_OR_NEAR_IDENTICAL",
             "Même photographie avec filigrane."),
        18: ("SIMILAR_AMBIGUOUS",
             "Montage de feuilles avec symptômes visuellement similaires."),
        19: ("SIMILAR_AMBIGUOUS",
             "Gros plan très similaire d'une feuille avec des pustules."),
        20: ("NOT_SAME_IMAGE",
             "Un épi de blé et un acarien isolé : contenus visuellement différents."),
        21: ("SAME_OR_NEAR_IDENTICAL",
             "Photographie très similaire à la paire 10.")
    }

    # ============================================================
    # 3. RESTAURER L'ORDRE ORIGINAL DES PLANCHES
    # ============================================================

    source = pd.read_csv(SOURCE_PAIRS)
    register = pd.read_csv(REGISTER_PATH)

    def make_key(row, col1, col2):
        return tuple(sorted([
            str(row[col1]).strip(),
            str(row[col2]).strip()
        ]))

    source["_key"] = source.apply(
        lambda r: make_key(r, "sha256_1", "sha256_2"),
        axis=1
    )

    register["_key"] = register.apply(
        lambda r: make_key(r, "sha256_1", "sha256_2"),
        axis=1
    )

    if source["_key"].duplicated().any():
        raise ValueError("Des paires répétées existent dans le fichier source.")

    if register["_key"].duplicated().any():
        raise ValueError("Des paires répétées existent dans le registre.")

    if set(source["_key"]) != set(register["_key"]):
        raise ValueError("Les paires du registre ne correspondent pas à la source.")

    # L'ordre de source correspond à l'ordre des planches
    order_map = {
        key: i + 1
        for i, key in enumerate(source["_key"])
    }

    register["visual_pair_id"] = register["_key"].map(order_map)

    register["visual_decision"] = register["visual_pair_id"].map(
        lambda i: observations[i][0]
    )

    register["visual_comment"] = register["visual_pair_id"].map(
        lambda i: observations[i][1]
    )

    register = register.sort_values("visual_pair_id").reset_index(drop=True)

    register = register.drop(columns=["_key"])

    register.to_csv(REGISTER_PATH, index=False)

    print("Registre corrigé et sauvegardé.")
    print("\nDécisions visuelles :")
    print(register["visual_decision"].value_counts())

    # ============================================================
    # 4. IDENTIFICATION DES GROUPES CONNECTÉS
    # ============================================================

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # Construire les connexions entre empreintes SHA
    for _, row in register.iterrows():
        union(str(row["sha256_1"]), str(row["sha256_2"]))

    # Construire les groupes
    components = {}

    for sha in parent:
        root = find(sha)
        components.setdefault(root, []).append(sha)

    groups = {
        root: members
        for root, members in components.items()
        if len(members) >= 2
    }

    # Associer les paires à leurs groupes
    pair_group = {}

    for group_id, members in enumerate(groups.values(), start=1):
        for sha in members:
            pair_group[sha] = f"G{group_id:02d}"

    register["group_id"] = register.apply(
        lambda r: pair_group.get(str(r["sha256_1"]), "")
        if pair_group.get(str(r["sha256_1"]), "")
           == pair_group.get(str(r["sha256_2"]), "")
        else "CHECK",
        axis=1
    )

    register.to_csv(REGISTER_PATH, index=False)

    # ============================================================
    # 5. EXPORTER LES MEMBRES DES GROUPES
    # ============================================================

    group_rows = []

    for group_id, members in enumerate(groups.values(), start=1):
        group_id = f"G{group_id:02d}"

        for sha in members:
            matching = register[
                (register["sha256_1"].astype(str) == sha)
                | (register["sha256_2"].astype(str) == sha)
            ]

            for _, row in matching.iterrows():
                if str(row["sha256_1"]) == sha:
                    image = row["image_1"]
                    label = row["current_class_1"]
                    split = row["current_split_1"]
                else:
                    image = row["image_2"]
                    label = row["current_class_2"]
                    split = row["current_split_2"]

                group_rows.append({
                    "group_id": group_id,
                    "sha256": sha,
                    "image": image,
                    "canonical_class": label,
                    "split": split
                })

    group_df = pd.DataFrame(group_rows).drop_duplicates(
        subset=["group_id", "sha256"]
    )

    GROUP_PATH = AUDIT_DIR / "cross_class_connected_groups.csv"
    group_df.to_csv(GROUP_PATH, index=False)

    # ============================================================
    # 6. RÉSUMÉ
    # ============================================================

    print("\n" + "=" * 65)
    print("RÉSUMÉ DES GROUPES")
    print("=" * 65)

    print("Nombre de paires :", len(register))
    print("Nombre de groupes connectés :", len(groups))
    print("Nombre d'images distinctes dans les groupes :",
          group_df["sha256"].nunique())

    print("\nDétail des groupes :")

    summary = (
        group_df.groupby("group_id")
        .agg(
            images=("sha256", "nunique"),
            classes=("canonical_class", lambda x: ", ".join(sorted(set(x)))),
            partitions=("split", lambda x: ", ".join(sorted(set(x))))
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    print("\nFichiers enregistrés :")
    print(REGISTER_PATH)
    print(GROUP_PATH)

    print("\nAucune image du dataset n'a été modifiée.")


def stage_09e_cross_class_decisions():
    """Consolidation des decisions par groupe et par image (cross_class_image_decisions.csv).
    
    Origine : cellule(s) 91 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/classification_v2/integrity_audit
    """

    # ══════════ cellule 91 ══════════


    # ============================================================
    # 1. CHEMINS
    # ============================================================

    ROOT = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/classification_v2/integrity_audit"
    )

    REGISTER_PATH = ROOT / "cross_class_manual_validation_register.csv"
    GROUPS_PATH = ROOT / "cross_class_connected_groups.csv"

    OUTPUT_DIR = ROOT / "final_visual_validation"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assert REGISTER_PATH.exists(), f"Registre introuvable : {REGISTER_PATH}"
    assert GROUPS_PATH.exists(), f"Groupes introuvables : {GROUPS_PATH}"

    # ============================================================
    # 2. DECISIONS VISUELLES CONSOLIDEES
    # ============================================================

    decisions = {
        "G01": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G02": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G03": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G04": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G05": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G06": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G07": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G08": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G09": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G10": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G11": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G12": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G13": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G14": ("SAME_OR_NEAR_IDENTICAL", "LABEL_CONFLICT", "QUARANTINE_GROUP"),
        "G15": ("NOT_SAME_IMAGE", "NO_CONFLICT_OBSERVED", "KEEP_BOTH"),
    }

    # ============================================================
    # 3. CHARGEMENT ET VERIFICATION DES COLONNES
    # ============================================================

    register = pd.read_csv(REGISTER_PATH, dtype=str).fillna("")
    groups = pd.read_csv(GROUPS_PATH, dtype=str).fillna("")

    print("Colonnes du registre :", register.columns.tolist())
    print("Colonnes des groupes :", groups.columns.tolist())

    required_register = {"sha256_1", "sha256_2"}
    required_groups = {"group_id", "sha256"}

    if not required_register.issubset(register.columns):
        raise ValueError(
            f"Colonnes manquantes dans le registre : "
        f"{required_register - set(register.columns)}"
        )

    if not required_groups.issubset(groups.columns):
        raise ValueError(
            f"Colonnes manquantes dans le fichier des groupes : "
        f"{required_groups - set(groups.columns)}"
        )

    # Vérifier que les 15 groupes attendus existent
    found_groups = set(groups["group_id"].unique())
    expected_groups = set(decisions.keys())

    if found_groups != expected_groups:
        raise ValueError(
            f"Groupes incohérents.\n"
        f"Manquants : {expected_groups - found_groups}\n"
        f"Inattendus : {found_groups - expected_groups}"
        )

    # ============================================================
    # 4. CONSTRUIRE LA CORRESPONDANCE SHA256 -> GROUPE
    # ============================================================

    membership = groups[["group_id", "sha256"]].drop_duplicates()

    # Une image ne doit appartenir qu'à un seul groupe
    duplicate_membership = membership[
        membership.duplicated("sha256", keep=False)
    ]

    if not duplicate_membership.empty:
        raise ValueError(
            "Certaines images appartiennent à plusieurs groupes :\n"
            + duplicate_membership.to_string(index=False)
        )

    sha_to_group = dict(zip(membership["sha256"], membership["group_id"]))

    # ============================================================
    # 5. RELIER LES 21 PAIRES AUX GROUPES
    # ============================================================

    register["group_id"] = ""

    for idx, row in register.iterrows():
        sha1 = row["sha256_1"]
        sha2 = row["sha256_2"]

        group1 = sha_to_group.get(sha1)
        group2 = sha_to_group.get(sha2)

        if group1 and group1 == group2:
            register.at[idx, "group_id"] = group1

    unmatched = register[register["group_id"] == ""]

    if not unmatched.empty:
        print("\nPaires non reliées à un groupe :")
        print(
            unmatched[
                ["sha256_1", "sha256_2"]
            ].to_string(index=False)
        )
        raise ValueError(
            f"{len(unmatched)} paires ne sont pas reliées à un groupe. "
        "Arrêt sans écraser le registre."
        )

    # ============================================================
    # 6. APPLIQUER LES DECISIONS AU REGISTRE
    # ============================================================

    for column in [
        "visual_decision",
        "annotation_decision",
        "final_action"
    ]:
        if column not in register.columns:
            register[column] = ""

    for idx, row in register.iterrows():
        group_id = row["group_id"]
        visual, annotation, action = decisions[group_id]

        register.at[idx, "visual_decision"] = visual
        register.at[idx, "annotation_decision"] = annotation
        register.at[idx, "final_action"] = action

        if "review_comment" in register.columns:
            register.at[idx, "review_comment"] = (
                f"Validation visuelle du groupe {group_id}. "
            "Décision basée sur la comparaison des planches. "
            "La classe correcte n'est pas déterminée."
            )

    # ============================================================
    # 7. REGISTRE DES DECISIONS PAR GROUPE
    # ============================================================

    group_rows = []

    for group_id, (visual, annotation, action) in decisions.items():
        subset = groups[groups["group_id"] == group_id]

        group_rows.append({
            "group_id": group_id,
            "number_of_images": subset["sha256"].nunique(),
            "visual_decision": visual,
            "annotation_decision": annotation,
            "final_action": action,
            "sha256_list": " | ".join(sorted(subset["sha256"].unique()))
        })

    group_decisions = pd.DataFrame(group_rows)

    # ============================================================
    # 8. REGISTRE DES IMAGES
    # ============================================================

    image_decisions = groups.drop_duplicates(
        ["group_id", "sha256"]
    ).copy()

    image_decisions["visual_decision"] = image_decisions["group_id"].map(
        {k: v[0] for k, v in decisions.items()}
    )

    image_decisions["annotation_decision"] = image_decisions["group_id"].map(
        {k: v[1] for k, v in decisions.items()}
    )

    image_decisions["final_action"] = image_decisions["group_id"].map(
        {k: v[2] for k, v in decisions.items()}
    )

    # ============================================================
    # 9. CONTROLES DE COHERENCE
    # ============================================================

    quarantine_groups = [
        group_id for group_id, decision in decisions.items()
        if decision[2] == "QUARANTINE_GROUP"
    ]

    keep_groups = [
        group_id for group_id, decision in decisions.items()
        if decision[2] == "KEEP_BOTH"
    ]

    quarantine_images = image_decisions[
        image_decisions["final_action"] == "QUARANTINE_GROUP"
    ]

    keep_images = image_decisions[
        image_decisions["final_action"] == "KEEP_BOTH"
    ]

    print("\n" + "=" * 65)
    print("CONTROLES DE VALIDATION")
    print("=" * 65)

    print("Nombre de groupes :", len(group_decisions))
    print("Groupes à mettre en quarantaine :", len(quarantine_groups))
    print("Groupes à conserver :", len(keep_groups))
    print("Images candidates à la quarantaine :", len(quarantine_images))
    print("Images à conserver :", len(keep_images))
    print("Paires manuelles reliées :", len(register))

    assert len(group_decisions) == 15
    assert len(quarantine_groups) == 14
    assert len(keep_groups) == 1
    assert len(quarantine_images) == 34
    assert len(keep_images) == 2
    assert set(keep_images["group_id"]) == {"G15"}

    # ============================================================
    # 10. SAUVEGARDE ET EXPORT
    # ============================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = OUTPUT_DIR / f"register_backup_{timestamp}.csv"
    shutil.copy2(REGISTER_PATH, backup_path)

    register_output = OUTPUT_DIR / "cross_class_manual_validation_register_final.csv"
    groups_output = OUTPUT_DIR / "cross_class_group_decisions.csv"
    images_output = OUTPUT_DIR / "cross_class_image_decisions.csv"

    register.to_csv(register_output, index=False)
    group_decisions.to_csv(groups_output, index=False)
    image_decisions.to_csv(images_output, index=False)

    print("\n" + "=" * 65)
    print("CONSOLIDATION TERMINEE")
    print("=" * 65)

    print("Registre final :", register_output)
    print("Décisions groupes :", groups_output)
    print("Décisions images :", images_output)
    print("Sauvegarde initiale :", backup_path)

    print("\nAucune image du dataset n'a été modifiée.")


def stage_10_build_classification_v3():
    """Construction de classification_v3 : v2 prive des images en QUARANTINE_GROUP (12 484 images). Contient la RECONSTRUCTION de `kept` (cellule d'origine absente).
    
    Origine : cellule(s) 92, 93, 94, 95, 96, 97 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : A VALIDER : le choix de cette version/ce role repose sur une deduction (voir CELL_MAP.md).
    ATTENTION : Contient un bloc RECONSTRUIT (variable `kept`).
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v2 ; data/processed/classification_v3 ; data/processed/classification_v3 ; data/processed/classification_v3
    """

    # ══════════ cellule 92 ══════════


    # ============================================================
    # 1. CHEMINS
    # ============================================================


    V2_DIR = PROJECT / "data/processed/classification_v2"
    V2_MANIFEST = V2_DIR / "classification_manifest.csv"

    FINAL_DIR = V2_DIR / "integrity_audit/final_visual_validation"
    IMAGE_DECISIONS = FINAL_DIR / "cross_class_image_decisions.csv"

    # ============================================================
    # 2. CHARGEMENT
    # ============================================================

    manifest = pd.read_csv(V2_MANIFEST, dtype=str).fillna("")
    decisions = pd.read_csv(IMAGE_DECISIONS, dtype=str).fillna("")

    # Vérification des colonnes
    required_manifest = {"sha256", "canonical_class", "new_split"}
    required_decisions = {
        "sha256", "group_id", "final_action",
        "canonical_class", "split"
    }

    assert required_manifest.issubset(manifest.columns)
    assert required_decisions.issubset(decisions.columns)

    assert manifest["sha256"].is_unique, (
        "Le manifeste contient des SHA256 dupliqués."
    )

    # ============================================================
    # 3. SEPARER LES DECISIONS
    # ============================================================

    quarantine = decisions[
        decisions["final_action"] == "QUARANTINE_GROUP"
    ].copy()

    keep = decisions[
        decisions["final_action"] == "KEEP_BOTH"
    ].copy()

    assert quarantine["sha256"].is_unique
    assert keep["sha256"].is_unique

    # ============================================================
    # 4. PREPARER LES METADONNEES DU MANIFESTE
    # ============================================================

    manifest_info = manifest[
        ["sha256", "canonical_class", "new_split"]
    ].rename(columns={
        "canonical_class": "audit_class",
        "new_split": "audit_split"
    })

    # ============================================================
    # 5. VERIFIER LA CORRESPONDANCE SHA256
    # ============================================================

    excluded_info = quarantine.merge(
        manifest_info,
        on="sha256",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    kept_info = keep.merge(
        manifest_info,
        on="sha256",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    missing_excluded = excluded_info[
        excluded_info["_merge"] != "both"
    ]

    missing_kept = kept_info[
        kept_info["_merge"] != "both"
    ]

    if not missing_excluded.empty or not missing_kept.empty:
        raise ValueError(
            "Certaines images sont absentes du manifeste V2."
        )

    excluded_info.drop(columns="_merge", inplace=True)
    kept_info.drop(columns="_merge", inplace=True)

    # ============================================================
    # 6. VERIFIER LES CLASSES ET PARTITIONS
    # ============================================================

    # Comparaison entre le registre visuel et le manifeste
    class_mismatch = excluded_info[
        excluded_info["canonical_class"] != excluded_info["audit_class"]
    ]

    split_mismatch = excluded_info[
        excluded_info["split"] != excluded_info["audit_split"]
    ]

    if not class_mismatch.empty:
        print("Attention : différences de classes :")
        print(
            class_mismatch[
                ["group_id", "sha256", "canonical_class", "audit_class"]
            ].to_string(index=False)
        )

    if not split_mismatch.empty:
        print("Attention : différences de partitions :")
        print(
            split_mismatch[
                ["group_id", "sha256", "split", "audit_split"]
            ].to_string(index=False)
        )

    # ============================================================
    # 7. CONTROLES FINAUX
    # ============================================================

    print("\n" + "=" * 65)
    print("AUDIT PREALABLE CLASSIFICATION_V3")
    print("=" * 65)

    print("Images dans V2 :", len(manifest))
    print("Images candidates à exclure :", len(excluded_info))
    print("Images de G15 à conserver :", len(kept_info))
    print("Images candidates introuvables :", len(missing_excluded))
    print("Images à conserver introuvables :", len(missing_kept))

    print("\nRépartition par partition :")
    print(
        excluded_info["audit_split"]
        .value_counts()
        .reindex(["train", "val", "test"], fill_value=0)
        .to_string()
    )

    print("\nRépartition par classe :")
    print(
        excluded_info["audit_class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDétail des images G15 à conserver :")
    print(
        kept_info[
            ["group_id", "sha256", "audit_split", "audit_class"]
        ].to_string(index=False)
    )

    # ============================================================
    # 8. ASSERTIONS
    # ============================================================

    assert len(manifest) == 12518
    assert len(excluded_info) == 34
    assert len(kept_info) == 2
    assert len(missing_excluded) == 0
    assert len(missing_kept) == 0

    assert set(kept_info["group_id"]) == {"G15"}

    assert class_mismatch.empty, (
        "Les classes du registre et du manifeste ne correspondent pas."
    )

    assert split_mismatch.empty, (
        "Les partitions du registre et du manifeste ne correspondent pas."
    )

    print("\n" + "=" * 65)
    print("AUDIT PREALABLE TERMINE AVEC SUCCES")
    print("Aucune image n'a été modifiée.")
    print("=" * 65)

    # ══════════ RECONSTRUIT : cellule d'origine ABSENTE du notebook ══════════
    # Les cellules 93-97 utilisent un DataFrame `kept` qu'AUCUNE cellule du notebook
    # ne definit. D'apres la cellule 92 (manifeste V2 = 12 518 images) et la
    # verification finale de v3 (12 484 images, cellule 98), `kept` est le manifeste
    # V2 prive des images en QUARANTINE_GROUP. A VALIDER si vous retrouvez la
    # cellule d'origine ; les assertions ci-dessous echouent si l'hypothese est fausse.
    kept = manifest[~manifest["sha256"].isin(quarantine["sha256"])].copy()
    assert len(kept) == 12484, f"kept reconstruit = {len(kept)} images (attendu 12484)"

    # ══════════ cellule 93 ══════════

    # Dossier de destination
    DEST_DIR = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    )

    # Génération de chemins uniques
    def build_unique_destination(row):
        split = str(row["new_split"])
        class_name = str(row["canonical_class"])
        original_name = Path(str(row["filename"])).name

        sha = str(row["sha256"]).lower()
        short_id = sha[:12]

        original_path = Path(original_name)

        new_filename = (
            f"{original_path.stem}_{short_id}"
        f"{original_path.suffix.lower()}"
        )

        return DEST_DIR / split / class_name / new_filename


    kept["v3_destination_path"] = kept.apply(
        build_unique_destination,
        axis=1
    )

    # Vérification des collisions
    collisions = kept[
        kept["v3_destination_path"].duplicated(keep=False)
    ]

    if not collisions.empty:
        raise ValueError(
            "Des collisions persistent :\n"
            + collisions[
                ["sha256", "v3_destination_path"]
            ].to_string(index=False)
        )

    print("Vérification des chemins terminée.")
    print(f"Images à copier : {len(kept)}")
    print(f"Nombre de collisions : {len(collisions)}")

    # ══════════ cellule 94 ══════════


    DEST_DIR = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    )

    # Destinations attendues
    expected_paths = {
        str(Path(p).resolve())
        for p in kept["v3_destination_path"]
    }

    # Fichiers réellement présents dans les dossiers de classes
    actual_paths = {
        str(p.resolve())
        for split in ["train", "val", "test"]
        for p in (DEST_DIR / split).rglob("*")
        if p.is_file()
    }

    missing_paths = sorted(expected_paths - actual_paths)
    unexpected_paths = sorted(actual_paths - expected_paths)

    print("========== DIAGNOSTIC V3 ==========")
    print("Destinations attendues :", len(expected_paths))
    print("Fichiers réellement présents :", len(actual_paths))
    print("Fichiers manquants :", len(missing_paths))
    print("Fichiers inattendus :", len(unexpected_paths))

    # Rechercher les doublons dans le manifeste
    duplicate_destinations = kept[
        kept["v3_destination_path"].duplicated(keep=False)
    ]

    print(
        "\nDestinations dupliquées dans le manifeste :",
        len(duplicate_destinations)
    )

    if missing_paths:
        print("\nExemples de fichiers manquants :")
        for p in missing_paths[:20]:
            print(p)

    if unexpected_paths:
        print("\nExemples de fichiers inattendus :")
        for p in unexpected_paths[:20]:
            print(p)

    # Répartition des fichiers manquants
    if missing_paths:
        missing_df = pd.DataFrame({"path": missing_paths})
        missing_df["split"] = missing_df["path"].apply(
            lambda p: Path(p).parts[-3]
        )
        print("\nRépartition des fichiers manquants :")
        print(missing_df["split"].value_counts())

    # ══════════ cellule 95 ══════════

    # Identifier les destinations partagées par plusieurs images
    duplicate_groups = (
        kept[
            kept["v3_destination_path"].duplicated(keep=False)
        ]
        .sort_values("v3_destination_path")
        .copy()
    )

    print("Nombre de lignes concernées :", len(duplicate_groups))
    print(
        "Nombre de destinations concernées :",
        duplicate_groups["v3_destination_path"].nunique()
    )

    # Afficher les premières collisions
    display(
        duplicate_groups[
            [
                "filename",
                "sha256",
                "canonical_class",
                "new_split",
                "v3_destination_path"
            ]
        ].head(30)
    )

    # Vérifier si les SHA-256 sont différents au sein des collisions
    collision_summary = (
        duplicate_groups.groupby("v3_destination_path")
        .agg(
            number_of_images=("sha256", "size"),
            unique_sha256=("sha256", "nunique"),
            classes=("canonical_class", lambda x: ", ".join(sorted(set(x)))),
            splits=("new_split", lambda x: ", ".join(sorted(set(x))))
        )
        .reset_index()
    )

    print("\nRésumé des collisions :")
    display(collision_summary.head(30))

    print(
        "\nDestinations contenant plusieurs SHA-256 :",
        (collision_summary["unique_sha256"] > 1).sum()
    )

    # ══════════ cellule 96 ══════════


    # 1. Vérifier l'unicité des SHA-256
    assert kept["sha256"].astype(str).str.lower().nunique() == len(kept), (
        "Attention : les SHA-256 ne sont pas uniques dans kept."
    )

    DEST_DIR = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    )

    # 2. Construire un nom unique à partir du SHA-256 complet
    def build_unique_destination(row):
        split = str(row["new_split"])
        class_name = str(row["canonical_class"])
        extension = Path(str(row["filename"])).suffix.lower()

        sha = str(row["sha256"]).lower()

        new_filename = f"{sha}{extension}"

        return str(
            DEST_DIR / split / class_name / new_filename
        )

    kept["v3_destination_path"] = kept.apply(
        build_unique_destination,
        axis=1
    )

    # 3. Contrôler l'unicité des destinations
    duplicate_count = kept["v3_destination_path"].duplicated().sum()

    print("Images dans le manifeste :", len(kept))
    print("Destinations uniques :", kept["v3_destination_path"].nunique())
    print("Collisions restantes :", duplicate_count)

    assert duplicate_count == 0

    # 4. Afficher quelques exemples
    display(
        kept[
            ["filename", "sha256", "canonical_class",
             "new_split", "v3_destination_path"]
        ].head(10)
    )

    print("\nChemins uniques validés.")

    # ══════════ cellule 97 ══════════


    # IMPORTANT : cette cellule remplace uniquement classification_v3.
    # classification_v1 et classification_v2 ne sont pas touchés.

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)


    copied = 0

    for _, row in kept.iterrows():

        source = None

        for col in [
            "destination_path_v2",
            "destination_path",
            "image_path",
            "path"
        ]:
            if col in kept.columns:
                candidate = Path(str(row[col]))
                if candidate.is_file():
                    source = candidate
                    break

        if source is None:
            raise FileNotFoundError(
                f"Image source introuvable : {row['filename']}"
            )

        destination = Path(row["v3_destination_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, destination)

        if sha256_file(destination) != str(row["sha256"]).lower():
            raise ValueError(
                f"SHA-256 incorrect : {destination}"
            )

        copied += 1

    print("Images copiées et vérifiées :", copied)

    # Sauvegarde du manifeste
    kept.to_csv(
        DEST_DIR / "classification_manifest.csv",
        index=False
    )

    print("Manifeste sauvegardé.")
    print("Classification_v3 reconstruite.")


def stage_11_verify_classification_v3():
    """Verification finale de v3 : SHA-256 uniques, 0 fuite inter-partitions, comptes par split.
    
    Origine : cellule(s) 98 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/classification_v3
    """

    # ══════════ cellule 98 ══════════


    ROOT = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    )
    MANIFEST = ROOT / "classification_manifest.csv"

    df = pd.read_csv(MANIFEST)


    # 1. Nombre de fichiers physiques
    physical_files = [
        p for split in ["train", "val", "test"]
        for p in (ROOT / split).rglob("*")
        if p.is_file()
    ]

    # 2. Répartition des partitions et des classes
    split_counts = df["new_split"].value_counts().to_dict()
    class_counts = df["canonical_class"].value_counts().sort_index()

    # 3. Vérification des fichiers et SHA-256
    missing_files = []
    hash_errors = []

    for _, row in df.iterrows():
        path = Path(row["v3_destination_path"])

        if not path.is_file():
            missing_files.append(str(path))
            continue

        actual_hash = sha256_file(path)
        expected_hash = str(row["sha256"]).lower()

        if actual_hash != expected_hash:
            hash_errors.append(str(path))

    # 4. Unicité des SHA-256
    unique_hashes = df["sha256"].astype(str).str.lower().nunique()

    # 5. Recherche de fuite exacte entre partitions
    hash_split = df.groupby(
        df["sha256"].astype(str).str.lower()
    )["new_split"].nunique()

    cross_split_hashes = hash_split[hash_split > 1]

    # 6. Vérifications finales
    expected_splits = {
        "train": 8761,
        "val": 1859,
        "test": 1864
    }

    print("========== AUDIT FINAL V3 ==========")
    print("Images dans le manifeste :", len(df))
    print("Fichiers physiques :", len(physical_files))
    print("Classes :", df["canonical_class"].nunique())
    print("SHA-256 uniques :", unique_hashes)
    print("Fichiers manquants :", len(missing_files))
    print("Erreurs SHA-256 :", len(hash_errors))
    print("Fuites SHA-256 inter-partitions :", len(cross_split_hashes))

    print("\nRépartition des partitions :")
    for split in ["train", "val", "test"]:
        print(f"{split}: {split_counts.get(split, 0)}")

    print("\nRépartition par classe :")
    display(class_counts.rename("total").to_frame())

    # 7. Assertions
    assert len(df) == 12484
    assert len(physical_files) == 12484
    assert df["canonical_class"].nunique() == 17
    assert split_counts == expected_splits
    assert unique_hashes == 12484
    assert len(missing_files) == 0
    assert len(hash_errors) == 0
    assert len(cross_split_hashes) == 0

    print("\n====================================")
    print("AUDIT FINAL : TOUTES LES VÉRIFICATIONS RÉUSSIES")
    print("====================================")


STAGES = {
    "00": stage_00_acquire_datasets,
    "01": stage_01_prepare_mendeley,
    "02": stage_02_global_audit,
    "04": stage_04_deduplicate_v1,
    "05": stage_05_build_classification_v1,
    "06": stage_06_phash_near_duplicate_audit,
    "07b": stage_07b_phash_pair_review_sheets,
    "07c": stage_07c_classify_phash_pairs,
    "07d": stage_07d_init_manual_review_register,
    "08": stage_08_build_classification_v2,
    "09a": stage_09a_integrity_split_audit,
    "09c": stage_09c_cross_class_visual_validation,
    "09d": stage_09d_cross_class_connected_groups,
    "09e": stage_09e_cross_class_decisions,
    "10": stage_10_build_classification_v3,
    "11": stage_11_verify_classification_v3,
}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--stage", default="all", help="numero(s) d'etape separes par des virgules, ou 'all'")
    ap.add_argument("--list", action="store_true", help="lister les etapes")
    a = ap.parse_args(argv)
    if a.list:
        for k, f in STAGES.items(): print(f"{k:>5}  {f.__name__}  -  {(f.__doc__ or '').strip().splitlines()[0]}")
        return
    wanted = list(STAGES) if a.stage == "all" else [x.strip() for x in a.stage.split(",")]
    for k in wanted:
        if k not in STAGES: raise SystemExit(f"Etape inconnue : {k} (voir --list)")
    for k in wanted:
        print(f"\n{'=' * 70}\n  ETAPE {k} : {STAGES[k].__name__}\n{'=' * 70}")
        STAGES[k]()


if __name__ == "__main__":
    main()
