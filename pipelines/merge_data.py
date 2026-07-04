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
    key = _normalize(raw_name)
    if key in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key]
    key2 = key.replace("_", " ")
    if key2 in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key2]
    canonical = raw_name.strip().replace(" ", "_").replace("-", "_")
    return canonical


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
    Ignore les dossiers wrapper (train/test/valid/...) et vides.

    Returns:
        Liste de (class_dir: Path) contenant au moins une image directe.
    """
    class_dirs = []

    def _recurse(d: Path, depth: int = 0):
        if depth > 6:
            return
        # Images directes dans ce dossier ?
        direct_images = [f for f in d.iterdir() if _is_image(f)]
        if direct_images:
            class_dirs.append(d)
            return
        # Sinon on descend dans les sous-dossiers non ignorés
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name.lower() not in {".ds_store", "__macosx"}:
                _recurse(sub, depth + 1)

    if root.exists():
        _recurse(root)

    return class_dirs


def _scan_dataset_smart(root: Path, source_name: str,
                          skip_dirs: set = None) -> list:
    """
    Scanner universel : trouve tous les dossiers de classes quelle que soit
    la profondeur de l'arborescence, et mappe leurs noms vers les classes
    canoniques. Les dossiers dans skip_dirs ne sont pas utilisés comme
    labels (mais on descend quand même dedans pour trouver les sous-classes).
    """
    if skip_dirs is None:
        skip_dirs = SKIP_DIRS

    records = []
    if not root.exists():
        print(f"    ⚠️  Dossier introuvable : {root}")
        return records

    class_dirs = _find_class_dirs(root)

    for cls_dir in class_dirs:
        raw_name = cls_dir.name
        # Si le dossier est un wrapper connu (train/test/...) on passe
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


# ─────────────────────────────────────────────────────────────
# Collecte par dataset
# ─────────────────────────────────────────────────────────────

def _collect_ds1(paths_cfg: dict) -> list:
    root = Path(paths_cfg.get("ds1", "data/raw/raw"))
    print(f"\n  DS1 — Wheat Plant Diseases : {root}")
    records = _scan_dataset_smart(root, "DS1_WheatPlantDiseases")
    print(f"    → {len(records)} images | "
          f"{len(set(r['canonical'] for r in records))} classes")
    return records


def _collect_ds2(paths_cfg: dict) -> list:
    root = Path(paths_cfg.get("ds2", "data/raw/MendeleyWheat"))
    print(f"\n  DS2 — Mendeley Bangladesh : {root}")
    records = _scan_dataset_smart(root, "DS2_Mendeley")
    print(f"    → {len(records)} images | "
          f"{len(set(r['canonical'] for r in records))} classes")
    return records


def _collect_ds3(paths_cfg: dict) -> list:
    root = Path(paths_cfg.get("ds3", "data/raw/YellowRust19"))
    print(f"\n  DS3 — Yellow Rust Hayit : {root}")
    # Ce dataset a souvent une structure : YELLOW-RUST-19/RAW/<severity>/<images>
    # Le scanner récursif descend jusqu'aux dossiers de sévérité
    records = _scan_dataset_smart(root, "DS3_YellowRust19",
                                   skip_dirs=SKIP_DIRS | {"raw", "yellow-rust-19",
                                                           "yellow_rust_19"})
    print(f"    → {len(records)} images | "
          f"{len(set(r['canonical'] for r in records))} classes")
    return records


def _collect_ds4(paths_cfg: dict) -> list:
    root = Path(paths_cfg.get("ds4", "data/raw/ZenodoWheatSmall"))
    print(f"\n  DS4 — Zenodo Wheat Disease Small : {root}")
    records = _scan_dataset_smart(root, "DS4_Zenodo",
                                   skip_dirs=SKIP_DIRS | {
                                       "wheat disease dataset",
                                       "wheat_disease_dataset"
                                   })
    print(f"    → {len(records)} images | "
          f"{len(set(r['canonical'] for r in records))} classes")
    return records


def _collect_ds5_yolo(paths_cfg: dict) -> dict:
    """DS5 Roboflow — format YOLO. Scan récursif de images/ et labels/."""
    root = Path(paths_cfg.get("ds5", "data/raw/RoboflowWheatRust"))
    print(f"\n  DS5 — Roboflow Wheat Rust (YOLO bboxes) : {root}")

    result = {"images": [], "labels": [], "root": str(root)}
    if not root.exists():
        print(f"    ⚠️  Dossier introuvable")
        return result

    # Chercher récursivement toutes les images
    all_imgs = [p for p in root.rglob("*") if _is_image(p)]

    for img_path in sorted(all_imgs):
        # Chercher le label correspondant dans labels/ au même niveau
        # Structure possible : images/train/x.jpg → labels/train/x.txt
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
    return unique


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
    src_root  = Path(ds5_result["root"])

    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_str, lbl_str in zip(ds5_result["images"], ds5_result["labels"]):
        img_path = Path(img_str)
        # Déterminer le split d'origine
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
    # Utilise la colonne 'label' (nom unifié après _build_classification_structure)
    label_col = "label"
    source_col = "source"

    print("\n" + "=" * 60)
    print("  RAPPORT DE FUSION — WheatAI-Merged Dataset")
    print("=" * 60)
    print(f"\n  Total images (après déduplication) : {len(merged_df)}")
    print(f"  Classes canoniques                  : {merged_df[label_col].nunique()}")
    print(f"  Split : {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    print(f"\n  {'Classe':<28} {'Total':>7}  {'Train':>7}  {'Val':>6}  {'Test':>6}")
    print(f"  {'-'*60}")
    for cls in sorted(merged_df[label_col].unique()):
        n  = (merged_df[label_col] == cls).sum()
        t  = (train_df[label_col]  == cls).sum()
        v  = (val_df[label_col]    == cls).sum()
        te = (test_df[label_col]   == cls).sum()
        print(f"  {cls:<28} {n:>7}  {t:>7}  {v:>6}  {te:>6}")

    print(f"\n  Contribution par source :")
    for src, n in merged_df[source_col].value_counts().items():
        pct = n / len(merged_df) * 100
        print(f"    {src:<38} {n:>6} images ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────

def build_merged_dataset(config: dict, copy_images: bool = False) -> dict:
    """
    Pipeline complet de fusion des 5 datasets vers WheatAI-Merged.

    Args:
        config      : configuration globale
        copy_images : True = copie physique / False = symlinks (recommandé)

    Returns:
        dict avec merged_df, train_df, val_df, test_df, label2idx,
              idx2label, class_weights, yolo_data_yaml, output_dir
    """
    print("\n" + "=" * 60)
    print("  FUSION DES DATASETS — WheatAI-Merged v2")
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collecte ──
    print("\n[1/6] Collecte des images de chaque source...")
    all_records = []
    all_records.extend(_collect_ds1(paths_cfg))
    all_records.extend(_collect_ds2(paths_cfg))
    all_records.extend(_collect_ds3(paths_cfg))
    all_records.extend(_collect_ds4(paths_cfg))
    ds5_result = _collect_ds5_yolo(paths_cfg)
    print(f"\n  Total brut : {len(all_records)} images")

    # ── 2. Déduplication ──
    print("\n[2/6] Déduplication...")
    all_records = _deduplicate(all_records)

    # ── 3. Structure classification ──
    print("\n[3/6] Organisation par classe...")
    merged_df = _build_classification_structure(all_records, output_dir, copy_images)
    label_names = sorted(merged_df["label"].unique().tolist())
    label2idx   = {name: i for i, name in enumerate(label_names)}
    idx2label   = {i: name for name, i in label2idx.items()}
    merged_df["label_idx"] = merged_df["label"].map(label2idx)
    print(f"  {len(merged_df)} images, {len(label_names)} classes canoniques")

    # Afficher les classes non mappées restantes
    unknowns = [c for c in label_names
                if c not in CANONICAL_CLASS_MAP.values()
                and c not in {"Black_Point", "Fusarium_Foot_Rot"}]
    if unknowns:
        print(f"\n  ⚠️  Classes non reconnues dans le mapping (à vérifier) :")
        for c in unknowns:
            n = (merged_df["label"] == c).sum()
            print(f"    {c} → {n} images")

    # ── 4. Split ──
    print("\n[4/6] Split stratifié (70/15/15)...")
    train_df, val_df, test_df = _stratified_split(
        merged_df, train_ratio, val_ratio, seed
    )
    train_df["label_idx"] = train_df["label"].map(label2idx)
    val_df["label_idx"]   = val_df["label"].map(label2idx)
    test_df["label_idx"]  = test_df["label"].map(label2idx)

    # ── 5. Class weights ──
    print("\n[5/6] Calcul des poids de classe...")
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

    # ── 6. Sauvegarde ──
    print("\n[6/6] Sauvegarde...")
    proc_dir = output_dir / "processed"
    proc_dir.mkdir(exist_ok=True)

    train_df.to_csv(proc_dir / "merged_train.csv", index=False)
    val_df.to_csv(proc_dir   / "merged_val.csv",   index=False)
    test_df.to_csv(proc_dir  / "merged_test.csv",  index=False)

    metadata = {
        "dataset_name" : "WheatAI-Merged",
        "num_classes"  : len(label_names),
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

    print(f"  ✅ merged_train.csv   ({len(train_df)} lignes)")
    print(f"  ✅ merged_val.csv     ({len(val_df)} lignes)")
    print(f"  ✅ merged_test.csv    ({len(test_df)} lignes)")
    print(f"  ✅ merged_metadata.json")

    # YOLO subset
    yolo_yaml = ""
    if ds5_result["images"]:
        print(f"\n  Construction du sous-dataset YOLO (DS5)...")
        yolo_yaml = _build_yolo_subset(ds5_result, output_dir, roboflow_classes)
        print(f"  ✅ YOLO data.yaml : {yolo_yaml}")
    else:
        print(f"\n  ℹ️  DS5 Roboflow absent — sous-dataset YOLO ignoré")

    _print_merge_report(merged_df, train_df, val_df, test_df)

    print(f"\n  📁 Dataset fusionné : {output_dir}")
    print("  ✅ Fusion terminée\n")

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