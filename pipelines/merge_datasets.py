"""
pipelines/merge_datasets.py
-----------------------------
Phase Merge — Construction du dataset unifié WheatAI-Merged.

Corrections v2 :
  - Scanner récursif intelligent : descend dans les sous-dossiers
    pour trouver les vrais dossiers de classes (contenant des images),
    quelle que soit la profondeur de l'arborescence.
  - DS5 Roboflow : scan récursif des images/ et labels/ à n'importe
    quelle profondeur.
  - Correction KeyError 'canonical' dans _print_merge_report.
  - Ratio max/min protégé contre la division par zéro.
"""

import json
import shutil
import hashlib
import yaml as yaml_lib
import pandas as pd
import numpy as np

from pathlib import Path

try:
    from PIL import Image   # optionnel — utilisé uniquement pour vérifier l'intégrité
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─────────────────────────────────────────────────────────────
# Mapping canonique
# ─────────────────────────────────────────────────────────────

CANONICAL_CLASS_MAP = {
    # DS1 — Wheat Plant Diseases
    "aphid"                   : "Aphid",
    "black rust"              : "Black_Rust",
    "black_rust"              : "Black_Rust",
    "blast"                   : "Blast",
    "brown rust"              : "Brown_Rust",
    "brown_rust"              : "Brown_Rust",
    "common root rot"         : "Common_Root_Rot",
    "common_root_rot"         : "Common_Root_Rot",
    "fusarium head blight"    : "Fusarium_Head_Blight",
    "fusarium_head_blight"    : "Fusarium_Head_Blight",
    "healthy"                 : "Healthy",
    "leaf blight"             : "Leaf_Blight",
    "leaf_blight"             : "Leaf_Blight",
    "mildew"                  : "Mildew",
    "mite"                    : "Mite",
    "septoria"                : "Septoria",
    "smut"                    : "Smut",
    "stem fly"                : "Stem_fly",
    "stem_fly"                : "Stem_fly",
    "tan spot"                : "Tan_spot",
    "tan_spot"                : "Tan_spot",
    "yellow rust"             : "Yellow_Rust",
    "yellow_rust"             : "Yellow_Rust",
    # DS2 — Mendeley Bangladesh
    "black point"             : "Black_Point",
    "black_point"             : "Black_Point",
    "fusarium foot rot"       : "Fusarium_Foot_Rot",
    "fusarium_foot_rot"       : "Fusarium_Foot_Rot",
    "healthy leaf"            : "Healthy",
    "healthy_leaf"            : "Healthy",
    "wheat blast"             : "Blast",
    "wheat_blast"             : "Blast",
    # DS3 — Yellow Rust Hayit (sévérités)
    "yellow rust healthy"     : "Healthy",
    "yellowrust_healthy"      : "Healthy",
    "yellowrust"              : "Yellow_Rust",
    "rust"                    : "Yellow_Rust",
    "stripe rust"             : "Yellow_Rust",
    "stripe_rust"             : "Yellow_Rust",
    # noms numériques de sévérité
    "0"                       : "Healthy",
    "1"                       : "Yellow_Rust",
    "2"                       : "Yellow_Rust",
    "3"                       : "Yellow_Rust",
    "4"                       : "Yellow_Rust",
    "5"                       : "Yellow_Rust",
    # DS4 — Zenodo Brown 2023
    "powdery mildew"          : "Mildew",
    "powdery_mildew"          : "Mildew",
    "leaf rust"               : "Brown_Rust",
    "leaf_rust"               : "Brown_Rust",
    "stem rust"               : "Black_Rust",
    "stem_rust"               : "Black_Rust",
    "septoria leaf blotch"    : "Septoria",
    "septoria_leaf_blotch"    : "Septoria",
    # variations communes
    "brown_rust_disease"      : "Brown_Rust",
    "yellow_rust_disease"     : "Yellow_Rust",
    "stem_rust_disease"       : "Black_Rust",

    "r"    : "Yellow_Rust",
    "mr"   : "Yellow_Rust",
    "mrms" : "Yellow_Rust",
    "ms"   : "Yellow_Rust",
    "s"    : "Yellow_Rust",
}

# Dossiers à ignorer (splits, wrappers, système)
SKIP_DIRS = {
    "train", "test", "valid", "validation", "val",
    "images", "labels", "annotations",
    "__macosx", ".ds_store", "raw", "augmented", "balanced",
}


def _normalize(name: str) -> str:
    return name.lower().strip().replace("-", "_").replace(" ", "_")


def _resolve_canonical(raw_name: str) -> str:
    """
    Convertit un nom de classe provenant d'un dataset
    vers une classe canonique.
    """

    key = _normalize(raw_name)

    # -------------------------------------------------
    # Supprimer les suffixes parasites
    # -------------------------------------------------

    for suffix in [
        "_train",
        "_test",
        "_valid",
        "_val"
    ]:
        if key.endswith(suffix):
            key = key[:-len(suffix)]

    # -------------------------------------------------
    # Quelques corrections fréquentes
    # -------------------------------------------------

    aliases = {

        "blackpoint": "black_point",

        "fusariumfootrot": "fusarium_foot_rot",

        "healthyleaf": "healthy_leaf",

        "leafblight": "leaf_blight",

        "wheatblast": "wheat_blast",

        "stemfly": "stem_fly",

        "tanspot": "tan_spot",

        "yellowrust": "yellow_rust",

        "brownrust": "brown_rust",

        "blackrust": "black_rust",

        "commonrootrot": "common_root_rot",

        "fusariumheadblight": "fusarium_head_blight",

        "powderymildew": "powdery_mildew",

        "leafrust": "leaf_rust",

        "stemrust": "stem_rust",

        "septorialeafblotch": "septoria_leaf_blotch",
    }

    key = aliases.get(key, key)

    # -------------------------------------------------

    if key in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key]

    key2 = key.replace("_", " ")

    if key2 in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key2]

    return raw_name.strip().replace(" ", "_").replace("-", "_")


IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG", ".bmp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix in IMG_EXT


def _image_hash(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read(8192)).hexdigest()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Scanner récursif intelligent
# ─────────────────────────────────────────────────────────────
 
def _find_class_dirs(root: Path) -> list:
    """
    Descend récursivement dans root pour trouver les dossiers qui
    contiennent DIRECTEMENT des images (= dossiers de classes).
    """
    class_dirs = []
 
    def _recurse(d: Path, depth: int = 0):
        if depth > 6:
            return
        direct_images = [f for f in d.iterdir() if _is_image(f)]
        if direct_images:
            class_dirs.append(d)
            return
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name.lower() not in {".ds_store", "__macosx"}:
                _recurse(sub, depth + 1)
 
    if root.exists():
        _recurse(root)
 
    return class_dirs
 
 
def _scan_dataset_smart(root: Path, source_name: str,
                          skip_dirs: set = None) -> list:
    """Scanner générique : trouve tous les dossiers de classes récursivement."""
    if skip_dirs is None:
        skip_dirs = SKIP_DIRS
 
    records = []
    if not root.exists():
        print(f"     Dossier introuvable : {root}")
        return records
 
    class_dirs = _find_class_dirs(root)
 
    for cls_dir in class_dirs:
        raw_name = cls_dir.name
        if _normalize(raw_name) in skip_dirs:
            continue
        canonical = _resolve_canonical(raw_name)
        images = [f for f in cls_dir.iterdir() if _is_image(f)]
        for img in images:
            records.append({
                "filepath" : str(img),
                "class_raw": raw_name,
                "canonical": canonical,
                "source"   : source_name,
            })
 
    return records
 
 
def _find_single_subfolder(root: Path, name_contains: list) -> "Path | None":
    """
    Cherche UN SEUL dossier dont le nom contient l'un des mots-clés donnés
    (insensible à la casse), à n'importe quelle profondeur sous root.
    Utilisé pour choisir une seule version d'un dataset qui en propose
    plusieurs (Original / Augmented / Split), afin d'éviter le double
    comptage et les fuites train/test entre versions dérivées les unes
    des autres.
    """
    if not root.exists():
        return None
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        name_lower = candidate.name.lower()
        if any(kw in name_lower for kw in name_contains):
            return candidate
    return None
 
 
# ─────────────────────────────────────────────────────────────
# Collecte par dataset
# ─────────────────────────────────────────────────────────────
 
def _collect_ds1(paths_cfg: dict) -> list:
    """
    DS1 — Wheat Plant Diseases.
    Structure : train/<Classe>/ + test/<classe>_test/ + valid/<classe>_valid/
    Ces 3 splits sont disjoints (pas de duplication) → scan complet OK.
    """
    root = Path(paths_cfg.get("ds1", "data/raw/data"))
    print(f"\n  DS1 — Wheat Plant Diseases : {root}")
    records = _scan_dataset_smart(root, "DS1_WheatPlantDiseases")
    classes = set(r["canonical"] for r in records)
    print(f"    → {len(records)} images | {len(classes)} classes")
    return records
 
 
def _collect_ds2(paths_cfg: dict) -> list:
    """
    DS2 — Mendeley Bangladesh.
    Structure réelle : Wheat Disease/{Original Dataset, Augmented Dataset,
    Split Dataset/{Train,Test,Validation}}/<Classe>/
 
    IMPORTANT : on n'utilise QUE "Original Dataset" (images réelles,
    1603 au total). "Augmented Dataset" et "Split Dataset" sont dérivés
    des mêmes photos (transformations / re-split) — les inclure en plus
    aurait créé un triple comptage et un risque de fuite train/test
    (une photo originale en test, sa version augmentée en train).
    """
    root = Path(paths_cfg.get("ds2", "data/raw/MendeleyWheat"))
    print(f"\n  DS2 — Mendeley Bangladesh : {root}")
    records = []
    if not root.exists():
        print("     Introuvable")
        return records
 
    original_dir = _find_single_subfolder(root, ["original"])
    if original_dir is None:
        print("     'Original Dataset' non trouvé — fallback 'Augmented Dataset'")
        original_dir = _find_single_subfolder(root, ["augmented"])
 
    if original_dir is None:
        print("     Aucune version reconnue — scan générique (risque de doublons)")
        records = _scan_dataset_smart(root, "DS2_Mendeley")
    else:
        print(f"    Utilisation de : {original_dir.relative_to(root)}")
        for cls_dir in sorted(original_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            canonical = _resolve_canonical(cls_dir.name)
            for img in cls_dir.iterdir():
                if _is_image(img):
                    records.append({
                        "filepath" : str(img),
                        "class_raw": cls_dir.name,
                        "canonical": canonical,
                        "source"   : "DS2_Mendeley",
                    })
 
    classes = set(r["canonical"] for r in records)
    print(f"    → {len(records)} images | {len(classes)} classes")
    return records
 
 
def _collect_ds3(paths_cfg: dict) -> list:
    """
    DS3 — Yellow Rust Hayit (YELLOW-RUST-19).
    Structure réelle : RAW/RAW/<code>/  +  YELLOW-RUST-19/YELLOW-RUST-19/<code>/
    Codes : 0=Healthy, MR/MRMS/MS/R/S=Yellow_Rust (sévérités croissantes)
 
    IMPORTANT : on n'utilise QUE "RAW" (images réelles, ~5421 au total).
    "YELLOW-RUST-19" est la version augmentée/balancée (2500/code, 15000
    au total) dérivée des mêmes photos — même raisonnement que DS2 :
    éviter double comptage et fuite train/test.
    """
    root = Path(paths_cfg.get("ds3", "data/raw/YellowRust19"))
    print(f"\n  DS3 — Yellow Rust Hayit : {root}")
    records = []
    if not root.exists():
        print("     Introuvable")
        return records
 
    # Chercher un dossier "RAW" contenant directement les codes de sévérité
    raw_dir = None
    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir() and candidate.name.upper() == "RAW":
            children = [c for c in candidate.iterdir() if c.is_dir()]
            has_codes = any(
                c.name.upper() in {"0", "MR", "MRMS", "MS", "R", "S"}
                for c in children
            )
            if has_codes:
                raw_dir = candidate
                break
 
    if raw_dir is None:
        print("     'RAW' non trouvé — fallback 'YELLOW-RUST-19' (augmenté)")
        for candidate in sorted(root.rglob("*")):
            if (candidate.is_dir()
                    and "yellow" in candidate.name.lower()
                    and "rust" in candidate.name.lower()):
                children = [c for c in candidate.iterdir() if c.is_dir()]
                has_codes = any(
                    c.name.upper() in {"0", "MR", "MRMS", "MS", "R", "S"}
                    for c in children
                )
                if has_codes:
                    raw_dir = candidate
                    break
 
    if raw_dir is None:
        print("     Structure non reconnue — scan générique (risque de doublons)")
        records = _scan_dataset_smart(
            root, "DS3_YellowRust19",
            skip_dirs=SKIP_DIRS | {"raw", "yellow_rust_19"}
        )
    else:
        print(f"    Utilisation de : {raw_dir.relative_to(root)}")
        for cls_dir in sorted(raw_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            code = cls_dir.name.upper()
            canonical = _resolve_canonical(cls_dir.name)
            n_before = len(records)
            for img in cls_dir.iterdir():
                if _is_image(img):
                    records.append({
                        "filepath" : str(img),
                        "class_raw": cls_dir.name,
                        "canonical": canonical,
                        "source"   : "DS3_YellowRust19",
                    })
            print(f"      {code:<6} → {canonical:<15} "
                  f"({len(records) - n_before} images)")
 
    classes = set(r["canonical"] for r in records)
    print(f"    → {len(records)} images | {len(classes)} classes")
    return records
 
 
def _collect_ds4(paths_cfg: dict) -> list:
    """DS4 — Zenodo Wheat Disease Small. Structure : Wheat Disease Dataset/<Classe>/"""
    root = Path(paths_cfg.get("ds4", "data/raw/ZenodoWheatSmall"))
    print(f"\n  DS4 — Zenodo Wheat Disease Small : {root}")
    records = _scan_dataset_smart(
        root, "DS4_Zenodo",
        skip_dirs=SKIP_DIRS | {"wheat disease dataset", "wheat_disease_dataset"}
    )
    classes = set(r["canonical"] for r in records)
    print(f"    → {len(records)} images | {len(classes)} classes")
    return records
 
 
def _collect_ds5_yolo(paths_cfg: dict) -> dict:
    """DS5 — Roboflow Wheat Rust (format YOLO, bboxes réelles)."""
    root = Path(paths_cfg.get("ds5", "data/raw/RoboflowWheatRust"))
    print(f"\n  DS5 — Roboflow Wheat Rust (YOLO bboxes) : {root}")
 
    result = {"images": [], "labels": [], "root": str(root)}
    if not root.exists():
        print(f"     Dossier introuvable")
        return result
 
    all_imgs = [p for p in root.rglob("*") if _is_image(p)]
 
    for img_path in sorted(all_imgs):
        try:
            rel = img_path.relative_to(root)
            parts = list(rel.parts)
            if "images" in parts:
                idx = parts.index("images")
                parts[idx] = "labels"
                label_path = root / Path(*parts).with_suffix(".txt")
            else:
                label_path = img_path.with_suffix(".txt")
        except Exception:
            label_path = img_path.with_suffix(".txt")
 
        result["images"].append(str(img_path))
        result["labels"].append(str(label_path) if label_path.exists() else None)
 
    annotated = sum(1 for l in result["labels"] if l is not None)
    print(f"    → {len(result['images'])} images | {annotated} annotées (bboxes)")
    return result
 
 
# ─────────────────────────────────────────────────────────────
# Déduplication
# ─────────────────────────────────────────────────────────────
 
def _deduplicate(records: list) -> list:
    seen = {}
    unique = []
    n_dupes = 0
    for rec in records:
        h = _image_hash(Path(rec["filepath"]))
        if h and h in seen:
            n_dupes += 1
        else:
            if h:
                seen[h] = rec["filepath"]
            unique.append(rec)
    if n_dupes:
        print(f"    {n_dupes} doublons supprimés → {len(unique)} images uniques")
    else:
        print(f"    Aucun doublon exact détecté ({len(unique)} images)")
    return unique
 
 
# ─────────────────────────────────────────────────────────────
# Plafonnement des classes surreprésentées (cap)
# ─────────────────────────────────────────────────────────────
 
def _cap_class_count(records: list, max_per_class: int, seed: int) -> list:
    """
    Plafonne le nombre d'images par classe canonique à max_per_class,
    par sous-échantillonnage aléatoire (sans remise). Les classes en
    dessous du seuil sont conservées intégralement.
    """
    rng = np.random.default_rng(seed)
 
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["canonical"], []).append(rec)
 
    capped = []
    print(f"\n  Plafonnement à {max_per_class} images/classe maximum :")
    for cls_name in sorted(by_class.keys()):
        items = by_class[cls_name]
        n_before = len(items)
        if n_before > max_per_class:
            idx = rng.choice(n_before, size=max_per_class, replace=False)
            items = [items[i] for i in idx]
            print(f"    {cls_name:<25} {n_before:>6} → {max_per_class:>6} "
                  f"(sous-échantillonné)")
        else:
            print(f"    {cls_name:<25} {n_before:>6} → {n_before:>6} (conservé)")
        capped.extend(items)
 
    return capped
 
 
# ─────────────────────────────────────────────────────────────
# Organisation classification
# ─────────────────────────────────────────────────────────────
 
def _build_classification_structure(records: list, output_dir: Path,
                                     copy_images: bool = False) -> pd.DataFrame:
    cls_root = output_dir / "classification"
    cls_root.mkdir(parents=True, exist_ok=True)
    for cls_name in set(r["canonical"] for r in records):
        (cls_root / cls_name).mkdir(exist_ok=True)
 
    rows = []
    for rec in records:
        src = Path(rec["filepath"])
        dst = cls_root / rec["canonical"] / f"{rec['source']}_{src.name}"
        if not dst.exists():
            try:
                if copy_images:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())
            except Exception:
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    continue
        rows.append({
            "filepath" : str(dst),
            "label"    : rec["canonical"],
            "source"   : rec["source"],
        })
    return pd.DataFrame(rows)
 
 
# ─────────────────────────────────────────────────────────────
# Split stratifié
# ─────────────────────────────────────────────────────────────
 
def _stratified_split(df: pd.DataFrame, train_ratio: float,
                       val_ratio: float, seed: int) -> tuple:
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=1.0 - train_ratio,
        stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["label"], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))
 
 
# ─────────────────────────────────────────────────────────────
# YOLO subset (DS5)
# ─────────────────────────────────────────────────────────────
 
def _build_yolo_subset(ds5_result: dict, output_dir: Path,
                        roboflow_classes: list) -> str:
    yolo_root = output_dir / "yolo"
    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
 
    for img_str, lbl_str in zip(ds5_result["images"], ds5_result["labels"]):
        img_path = Path(img_str)
        parts_lower = [p.lower() for p in img_path.parts]
        if "test" in parts_lower:
            out_split = "test"
        elif "valid" in parts_lower or "val" in parts_lower:
            out_split = "val"
        else:
            out_split = "train"
 
        dst_img = yolo_root / "images" / out_split / img_path.name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except Exception:
                shutil.copy2(img_path, dst_img)
 
        if lbl_str:
            lbl_path = Path(lbl_str)
            dst_lbl = yolo_root / "labels" / out_split / lbl_path.name
            if lbl_path.exists() and not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)
 
    data_yaml = {
        "path"  : str(yolo_root.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : len(roboflow_classes),
        "names" : roboflow_classes,
    }
    yaml_path = yolo_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml_lib.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    return str(yaml_path)
 
 
# ─────────────────────────────────────────────────────────────
# Rapport
# ─────────────────────────────────────────────────────────────
 
def _print_merge_report(merged_df: pd.DataFrame, train_df: pd.DataFrame,
                         val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  RAPPORT DE FUSION — WheatAI-Merged Dataset")
    print("=" * 65)
    print(f"\n  Total images : {len(merged_df)}")
    print(f"  Classes      : {merged_df['label'].nunique()}")
    print(f"  Split        : {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
 
    print(f"\n  {'Classe':<28} {'Total':>7}  {'Train':>7}  {'Val':>6}  {'Test':>6}")
    print(f"  {'-'*62}")
    for cls in sorted(merged_df["label"].unique()):
        n  = (merged_df["label"] == cls).sum()
        t  = (train_df["label"]  == cls).sum()
        v  = (val_df["label"]    == cls).sum()
        te = (test_df["label"]   == cls).sum()
        print(f"  {cls:<28} {n:>7}  {t:>7}  {v:>6}  {te:>6}")
 
    print(f"\n  Contribution par source :")
    for src, n in merged_df["source"].value_counts().items():
        pct = n / len(merged_df) * 100
        print(f"    {src:<38} {n:>6} images ({pct:.1f}%)")
 
 
# ─────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────
 
def build_merged_dataset(config: dict, copy_images: bool = False) -> dict:
    """
    Pipeline complet de fusion des 5 datasets vers WheatAI-Merged.
 
    Args:
        config      : configuration globale (config.yaml)
        copy_images : True = copie physique / False = symlinks (défaut)
 
    Returns:
        dict avec merged_df, train_df, val_df, test_df, label2idx,
              idx2label, class_weights, yolo_data_yaml, output_dir
    """
    print("\n" + "=" * 60)
    print("  FUSION DES DATASETS — WheatAI-Merged v4")
    print("=" * 60)
 
    merge_cfg        = config.get("merge", {})
    paths_cfg        = merge_cfg.get("paths", {})
    seed             = config["project"]["seed"]
    train_ratio      = config["split"]["train_ratio"]
    val_ratio        = config["split"]["val_ratio"]
    output_dir       = Path(merge_cfg.get("output", "data/merged"))
    roboflow_classes = merge_cfg.get(
        "roboflow_classes", ["Yellow_Rust", "Brown_Rust", "Stem_Rust"]
    )
    max_per_class    = merge_cfg.get("max_images_per_class", None)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # ── 1. Collecte ──
    print("\n[1/7] Collecte des images de chaque source...")
    all_records = []
    all_records.extend(_collect_ds1(paths_cfg))
    all_records.extend(_collect_ds2(paths_cfg))
    all_records.extend(_collect_ds3(paths_cfg))
    all_records.extend(_collect_ds4(paths_cfg))
    ds5_result = _collect_ds5_yolo(paths_cfg)
    print(f"\n  Total brut : {len(all_records)} images")
 
    # ── 2. Déduplication ──
    print("\n[2/7] Déduplication...")
    all_records = _deduplicate(all_records)
 
    # ── 3. Plafonnement ──
    if max_per_class:
        print(f"\n[3/7] Plafonnement des classes (max {max_per_class}/classe)...")
        all_records = _cap_class_count(all_records, max_per_class, seed)
        print(f"\n  Total après plafonnement : {len(all_records)} images")
    else:
        print(f"\n[3/7] Plafonnement désactivé (max_images_per_class non défini)")
 
    # ── 4. Structure classification ──
    print("\n[4/7] Organisation par classe...")
    merged_df = _build_classification_structure(all_records, output_dir, copy_images)
    label_names = sorted(merged_df["label"].unique().tolist())
    label2idx   = {name: i for i, name in enumerate(label_names)}
    idx2label   = {i: name for name, i in label2idx.items()}
    merged_df["label_idx"] = merged_df["label"].map(label2idx)
    print(f"  {len(merged_df)} images, {len(label_names)} classes canoniques")
    print(f"  Classes : {label_names}")
 
    # ── 5. Split ──
    print("\n[5/7] Split stratifié (70/15/15)...")
    train_df, val_df, test_df = _stratified_split(
        merged_df, train_ratio, val_ratio, seed
    )
    for df_ in (train_df, val_df, test_df):
        df_["label_idx"] = df_["label"].map(label2idx)
 
    # ── 6. Poids de classe ──
    print("\n[6/7] Calcul des poids de classe...")
    from sklearn.utils.class_weight import compute_class_weight
    cw_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_names)),
        y=train_df["label_idx"].values
    )
    class_weights = {i: float(w) for i, w in enumerate(cw_array)}
    cw_vals = list(class_weights.values())
    ratio = max(cw_vals) / min(cw_vals) if min(cw_vals) > 0 else float("inf")
    print(f"  Ratio max/min : {ratio:.2f}x")
 
    # ── 7. Sauvegarde ──
    print("\n[7/7] Sauvegarde...")
    proc_dir = output_dir / "processed"
    proc_dir.mkdir(exist_ok=True)
 
    train_df.to_csv(proc_dir / "merged_train.csv", index=False)
    val_df.to_csv(proc_dir   / "merged_val.csv",   index=False)
    test_df.to_csv(proc_dir  / "merged_test.csv",  index=False)
 
    metadata = {
        "dataset_name" : "WheatAI-Merged",
        "version"      : "v4",
        "num_classes"  : len(label_names),
        "max_images_per_class": max_per_class,
        "label2idx"    : label2idx,
        "idx2label"    : {str(k): v for k, v in idx2label.items()},
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "splits": {
            "train": len(train_df),
            "val"  : len(val_df),
            "test" : len(test_df),
            "total": len(merged_df),
        },
        "seed": seed,
    }
    with open(proc_dir / "merged_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
 
    print(f"  merged_train.csv   ({len(train_df)} lignes)")
    print(f"  merged_val.csv     ({len(val_df)} lignes)")
    print(f"  merged_test.csv    ({len(test_df)} lignes)")
    print(f"  merged_metadata.json")
 
    # YOLO
    yolo_yaml = ""
    if ds5_result["images"]:
        print(f"\n  Construction sous-dataset YOLO (DS5)...")
        yolo_yaml = _build_yolo_subset(ds5_result, output_dir, roboflow_classes)
        print(f"  YOLO data.yaml : {yolo_yaml}")
    else:
        print(f"\n   DS5 Roboflow absent — YOLO ignoré")
 
    _print_merge_report(merged_df, train_df, val_df, test_df)
 
    print(f"\n  Dataset fusionné : {output_dir}")
    print("  Fusion terminée\n")
 
    return {
        "merged_df"     : merged_df,
        "train_df"      : train_df,
        "val_df"        : val_df,
        "test_df"       : test_df,
        "label2idx"     : label2idx,
        "idx2label"     : idx2label,
        "class_weights" : class_weights,
        "yolo_data_yaml": yolo_yaml,
        "output_dir"    : str(output_dir),
    }
 