"""
pipelines/merge_datasets.py
-----------------------------
Phase 1+ — Construction du dataset unifié "WheatAI-Merged".

Fusionne 5 datasets de maladies du blé en un seul dataset propre,
re-splitté et prêt pour la classification ET la détection YOLO.

Datasets sources :
  DS1 — Wheat Plant Diseases (Kaggle kushagra3204)       : 13 104 img, 15 classes
  DS2 — Mendeley Bangladesh (5gc7hwydwg)                 : 5 000 img,  5 classes
  DS3 — Yellow Rust Hayit / YELLOW-RUST-19 (Kaggle)     : ~15 000 img, 2 classes
  DS4 — Wheat Disease Small — Zenodo/Kaggle (Brown 2023) : 999 img,    5 classes
  DS5 — Wheat Rust Roboflow (bboxes YOLO format)        : variable,   3 classes

Sortie :
  data/merged/
  ├── classification/
  │   ├── <Classe>/          ← toutes images fusionnées par classe
  │   └── ...
  ├── processed/
  │   ├── merged_train.csv
  │   ├── merged_val.csv
  │   ├── merged_test.csv
  │   └── merged_metadata.json
  └── yolo/                   ← uniquement DS5 (vraies bboxes)
      ├── images/train|val|test/
      ├── labels/train|val|test/
      └── data.yaml

Stratégie de mapping des classes :
  Chaque dataset a ses propres noms de dossiers → on les mappe
  vers les 15 classes canoniques de ton dataset principal.
  Les classes nouvelles (ex: Black Point, Fusarium Foot Rot)
  sont ajoutées en tant que nouvelles classes.

Usage :
    from pipelines.merge_datasets import build_merged_dataset
    result = build_merged_dataset(config)
"""

import json
import shutil
import hashlib
import yaml as yaml_lib
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Mapping des classes — noms bruts → classe canonique
# ─────────────────────────────────────────────────────────────

# Classe canonique = le nom normalisé utilisé dans TOUT le pipeline.
# Chaque entrée : "nom_de_dossier_source_en_minuscules" -> "Classe_Canonique"
# Les noms sont mis en minuscules + underscores avant comparaison.

CANONICAL_CLASS_MAP = {
    # ── DS1 Wheat Plant Diseases (Kaggle) ─────────────────────
    "aphid"                : "Aphid",
    "black rust"           : "Black_Rust",
    "black_rust"           : "Black_Rust",
    "blast"                : "Blast",
    "brown rust"           : "Brown_Rust",
    "brown_rust"           : "Brown_Rust",
    "common root rot"      : "Common_Root_Rot",
    "common_root_rot"      : "Common_Root_Rot",
    "fusarium head blight" : "Fusarium_Head_Blight",
    "fusarium_head_blight" : "Fusarium_Head_Blight",
    "healthy"              : "Healthy",
    "leaf blight"          : "Leaf_Blight",
    "leaf_blight"          : "Leaf_Blight",
    "mildew"               : "Mildew",
    "mite"                 : "Mite",
    "septoria"             : "Septoria",
    "smut"                 : "Smut",
    "stem fly"             : "Stem_fly",
    "stem_fly"             : "Stem_fly",
    "tan spot"             : "Tan_spot",
    "tan_spot"             : "Tan_spot",
    "yellow rust"          : "Yellow_Rust",
    "yellow_rust"          : "Yellow_Rust",

    # ── DS2 Mendeley Bangladesh ───────────────────────────────
    "black point"          : "Black_Point",     # nouvelle classe
    "black_point"          : "Black_Point",
    "fusarium foot rot"    : "Fusarium_Foot_Rot",  # nouvelle classe
    "fusarium_foot_rot"    : "Fusarium_Foot_Rot",
    "healthy leaf"         : "Healthy",
    "healthy_leaf"         : "Healthy",
    "leaf blight"          : "Leaf_Blight",
    "wheat blast"          : "Blast",
    "wheat_blast"          : "Blast",

    # ── DS3 Yellow Rust Hayit (YELLOW-RUST-19) ────────────────
    # Contient 2 catégories : healthy + yellow rust
    # Les sous-dossiers de sévérité sont fusionnés dans Yellow_Rust
    "yellow rust healthy"  : "Healthy",
    "yellowrust_healthy"   : "Healthy",
    "0"                    : "Healthy",       # certaines versions numérotent
    "1"                    : "Yellow_Rust",   # sévérité 1
    "2"                    : "Yellow_Rust",
    "3"                    : "Yellow_Rust",
    "4"                    : "Yellow_Rust",
    "5"                    : "Yellow_Rust",
    "yellow_rust_1"        : "Yellow_Rust",
    "yellow_rust_2"        : "Yellow_Rust",
    "yellow_rust_3"        : "Yellow_Rust",
    "yellow_rust_4"        : "Yellow_Rust",
    "yellow_rust_5"        : "Yellow_Rust",
    "yellowrust"           : "Yellow_Rust",
    "yellowrust_1"         : "Yellow_Rust",
    "yellowrust_2"         : "Yellow_Rust",
    "yellowrust_3"         : "Yellow_Rust",
    "yellowrust_4"         : "Yellow_Rust",
    "yellowrust_5"         : "Yellow_Rust",
    "rust"                 : "Yellow_Rust",
    "stripe rust"          : "Yellow_Rust",
    "stripe_rust"          : "Yellow_Rust",

    # ── DS4 Wheat Disease Small (Zenodo / Brown 2023) ─────────
    # Classes : yellow_rust, brown_rust, septoria, mildew, healthy
    # (noms déjà proches des classes canoniques)
    "powdery mildew"       : "Mildew",
    "powdery_mildew"       : "Mildew",
    "leaf rust"            : "Brown_Rust",
    "leaf_rust"            : "Brown_Rust",
    "stem rust"            : "Black_Rust",
    "stem_rust"            : "Black_Rust",
    "septoria leaf blotch" : "Septoria",
    "septoria_leaf_blotch" : "Septoria",

    # ── DS5 Roboflow (classes de détection) ──────────────────
    # Géré séparément dans la section YOLO (bboxes conservées)
    "yellow_rust_roboflow" : "Yellow_Rust",
    "brown_rust_roboflow"  : "Brown_Rust",
    "stem_rust_roboflow"   : "Black_Rust",
}


def _normalize_class_name(raw_name: str) -> str:
    """Normalise un nom de dossier → clé de lookup dans CANONICAL_CLASS_MAP."""
    return raw_name.lower().strip().replace("-", "_").replace(" ", "_")


def _resolve_canonical(raw_name: str) -> str:
    """
    Retourne la classe canonique correspondant à un nom de dossier brut.
    Si aucun mapping n'est trouvé, nettoie et retourne le nom tel quel
    (préfixé d'un avertissement dans les logs).
    """
    key = _normalize_class_name(raw_name)
    if key in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key]
    # Essai sans underscores
    key2 = key.replace("_", " ")
    if key2 in CANONICAL_CLASS_MAP:
        return CANONICAL_CLASS_MAP[key2]
    # Fallback : retourner nettoyé avec avertissement
    canonical = raw_name.replace(" ", "_").replace("-", "_")
    print(f"    Classe non mappée : '{raw_name}' → conservée comme '{canonical}'")
    return canonical


# ─────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG", ".bmp", ".BMP"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix in IMG_EXTENSIONS


def _image_hash(path: Path) -> str:
    """MD5 des premiers 8KB de l'image — détection de doublons rapide."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read(8192)).hexdigest()
    except Exception:
        return ""


def _scan_flat_dataset(root: Path, source_name: str) -> list:
    """
    Scanne un dataset organisé en dossiers de classes (structure flat) :
      root/
        ClassName1/img1.jpg ...
        ClassName2/img2.jpg ...

    Returns:
        List of dicts {filepath, class_raw, canonical, source}
    """
    records = []
    if not root.exists():
        print(f"    Dossier introuvable : {root}")
        return records

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        canonical = _resolve_canonical(cls_dir.name)
        for img in cls_dir.rglob("*"):
            if _is_image(img):
                records.append({
                    "filepath" : str(img),
                    "class_raw": cls_dir.name,
                    "canonical": canonical,
                    "source"   : source_name,
                })
    return records


def _scan_split_dataset(root: Path, source_name: str,
                         split_dirs: dict = None) -> list:
    """
    Scanne un dataset déjà splitté en train/val/test :
      root/
        train/ClassName1/...
        val/ClassName1/...
        test/ClassName1/...

    split_dirs : ex. {"train": "train", "val": "valid", "test": "test"}
    """
    if split_dirs is None:
        split_dirs = {"train": "train", "val": "val", "test": "test"}

    records = []
    for split_name, split_folder in split_dirs.items():
        split_path = root / split_folder
        if not split_path.exists():
            continue
        for cls_dir in sorted(split_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            canonical = _resolve_canonical(cls_dir.name)
            for img in cls_dir.rglob("*"):
                if _is_image(img):
                    records.append({
                        "filepath" : str(img),
                        "class_raw": cls_dir.name,
                        "canonical": canonical,
                        "source"   : source_name,
                        "orig_split": split_name,
                    })
    return records


# ─────────────────────────────────────────────────────────────
# Collecte de chaque dataset source
# ─────────────────────────────────────────────────────────────

def _collect_ds1_wheat_plant_diseases(paths_cfg: dict) -> list:
    """DS1 — Wheat Plant Diseases (Kaggle kushagra3204) — structure flat."""
    root = Path(paths_cfg.get("ds1", "data/raw/WheatPlantDiseases"))
    print(f"\n  DS1 — Wheat Plant Diseases : {root}")
    records = _scan_flat_dataset(root, "DS1_WheatPlantDiseases")
    print(f"    → {len(records)} images trouvées")
    return records


def _collect_ds2_mendeley(paths_cfg: dict) -> list:
    """DS2 — Mendeley Bangladesh — structure splitté train/test/val."""
    root = Path(paths_cfg.get("ds2", "data/raw/Mendeley_Bangladesh"))
    print(f"\n  DS2 — Mendeley Bangladesh : {root}")
    # Ce dataset a train/test/validation comme splits
    records = _scan_split_dataset(
        root, "DS2_Mendeley",
        split_dirs={"train": "train", "val": "validation", "test": "test"}
    )
    if not records:
        # Essai structure flat
        records = _scan_flat_dataset(root, "DS2_Mendeley")
    print(f"    → {len(records)} images trouvées")
    return records


def _collect_ds3_yellow_rust_hayit(paths_cfg: dict) -> list:
    """DS3 — Yellow Rust Hayit (YELLOW-RUST-19) — structure avec sous-dossiers de sévérité."""
    root = Path(paths_cfg.get("ds3", "data/raw/YellowRust19"))
    print(f"\n  DS3 — Yellow Rust Hayit : {root}")
    records = []

    if not root.exists():
        print(f"      Dossier introuvable : {root}")
        return records

    # Ce dataset peut avoir :
    # Structure A : root/healthy/*.jpg  root/yellow_rust_1/*.jpg  ...
    # Structure B : root/train/healthy/  root/train/yellowrust/  ...
    # On essaie les deux
    records = _scan_flat_dataset(root, "DS3_YellowRust19")
    if not records:
        records = _scan_split_dataset(root, "DS3_YellowRust19")

    # Correction du mapping sévérité → Yellow_Rust déjà géré dans CANONICAL_CLASS_MAP
    print(f"    → {len(records)} images trouvées")
    return records


def _collect_ds4_zenodo_brown(paths_cfg: dict) -> list:
    """DS4 — Wheat Disease Small (Zenodo / Brown 2023) — structure flat."""
    root = Path(paths_cfg.get("ds4", "data/raw/ZenodoWheatSmall"))
    print(f"\n  DS4 — Zenodo Wheat Disease Small : {root}")
    records = _scan_flat_dataset(root, "DS4_Zenodo")
    if not records:
        records = _scan_split_dataset(root, "DS4_Zenodo")
    print(f"    → {len(records)} images trouvées")
    return records


def _collect_ds5_roboflow_yolo(paths_cfg: dict) -> dict:
    """
    DS5 — Wheat Rust Roboflow (format YOLO avec bboxes).
    Ce dataset est traité SÉPARÉMENT des autres :
    - Les images sont copiées dans data/merged/yolo/
    - Les fichiers .txt de labels (bboxes) sont conservés
    - On retourne les infos pour la construction du data.yaml YOLO fusionné
    """
    root = Path(paths_cfg.get("ds5", "data/raw/RoboflowWheatRust"))
    print(f"\n  DS5 — Roboflow Wheat Rust (YOLO bboxes) : {root}")

    result = {"images": [], "labels": [], "root": str(root)}

    if not root.exists():
        print(f"      Dossier introuvable : {root}")
        return result

    # Structure YOLO standard : images/train|val|test + labels/train|val|test
    for split in ["train", "valid", "val", "test"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.rglob("*")):
            if _is_image(img_path):
                label_path = lbl_dir / (img_path.stem + ".txt")
                result["images"].append(str(img_path))
                result["labels"].append(str(label_path) if label_path.exists() else None)

    print(f"    → {len(result['images'])} images avec bboxes trouvées")
    return result


# ─────────────────────────────────────────────────────────────
# Déduplication
# ─────────────────────────────────────────────────────────────

def _deduplicate(records: list, verbose: bool = True) -> list:
    """
    Supprime les doublons par hash MD5 des 8 premiers KB de chaque image.
    Garde la première occurrence trouvée (DS1 en priorité).
    """
    seen_hashes = {}
    unique = []
    n_dupes = 0

    for rec in records:
        h = _image_hash(Path(rec["filepath"]))
        if h and h in seen_hashes:
            n_dupes += 1
        else:
            if h:
                seen_hashes[h] = rec["filepath"]
            unique.append(rec)

    if verbose and n_dupes > 0:
        print(f"   Déduplication : {n_dupes} doublons supprimés → {len(unique)} images uniques")

    return unique


# ─────────────────────────────────────────────────────────────
# Organisation et split
# ─────────────────────────────────────────────────────────────

def _build_classification_structure(records: list, output_dir: Path,
                                     copy_images: bool = False) -> pd.DataFrame:
    """
    Organise toutes les images dans output_dir/classification/<Classe>/.
    Par défaut crée des symlinks (économie d'espace), sinon copie.

    Returns:
        DataFrame avec colonnes filepath (source), canonical, source
    """
    cls_root = output_dir / "classification"
    cls_root.mkdir(parents=True, exist_ok=True)

    for cls_name in set(r["canonical"] for r in records):
        (cls_root / cls_name).mkdir(exist_ok=True)

    final_records = []
    for rec in records:
        src = Path(rec["filepath"])
        dst_dir = cls_root / rec["canonical"]
        dst = dst_dir / f"{rec['source']}_{src.name}"

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

        final_records.append({
            "filepath" : str(dst),
            "canonical": rec["canonical"],
            "source"   : rec["source"],
        })

    return pd.DataFrame(final_records)


def _stratified_split_df(df: pd.DataFrame, train_ratio: float,
                          val_ratio: float, seed: int) -> tuple:
    """Split stratifié sur la colonne 'canonical'."""
    from sklearn.model_selection import train_test_split

    label_col = "canonical"
    train_df, temp_df = train_test_split(
        df, test_size=1.0 - train_ratio,
        stratify=df[label_col], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df[label_col], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────
# Construction du sous-dataset YOLO (DS5 uniquement)
# ─────────────────────────────────────────────────────────────

def _build_yolo_subset(ds5_result: dict, output_dir: Path,
                        roboflow_classes: list) -> str:
    """
    Copie les images + labels du dataset Roboflow dans data/merged/yolo/
    et génère le data.yaml correspondant.

    Returns:
        Chemin vers le data.yaml généré
    """
    yolo_root = output_dir / "yolo"
    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    src_root = Path(ds5_result["root"])

    for split in ["train", "valid", "val", "test"]:
        out_split = "val" if split == "valid" else split
        img_src = src_root / "images" / split
        lbl_src = src_root / "labels" / split
        if not img_src.exists():
            continue

        for img_path in sorted(img_src.rglob("*")):
            if not _is_image(img_path):
                continue
            dst_img = yolo_root / "images" / out_split / img_path.name
            dst_lbl = yolo_root / "labels" / out_split / (img_path.stem + ".txt")

            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except Exception:
                    shutil.copy2(img_path, dst_img)

            lbl_src_path = lbl_src / (img_path.stem + ".txt")
            if lbl_src_path.exists() and not dst_lbl.exists():
                shutil.copy2(lbl_src_path, dst_lbl)

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
# Rapport de fusion
# ─────────────────────────────────────────────────────────────

def _print_merge_report(df: pd.DataFrame,
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame) -> None:

    counts = df["label"].value_counts().sort_index()
    sources = df["source"].value_counts()

    print("\n" + "=" * 60)
    print("  RAPPORT DE FUSION — WheatAI-Merged Dataset")
    print("=" * 60)

    print(f"\n  Total images (après déduplication) : {len(df)}")
    print(f"  Nombre de classes canoniques       : {df['label'].nunique()}")

    print(f"\n  Split : {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    print(f"\n  {'Classe':<28} {'Images':>8} {'Train':>7} {'Val':>6} {'Test':>6}")
    print("-" * 65)

    for cls in sorted(df["label"].unique()):
        n = (df["label"] == cls).sum()
        t = (train_df["label"] == cls).sum()
        v = (val_df["label"] == cls).sum()
        te = (test_df["label"] == cls).sum()

        print(f"{cls:<28} {n:>8} {t:>7} {v:>6} {te:>6}")

    print("\nContribution par source :")

    for src, n in sources.items():
        print(f"  {src:<35} {n:>6} images ({100*n/len(df):.1f}%)")

# ─────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────

def build_merged_dataset(config: dict,
                          copy_images: bool = False) -> dict:
    """
    Pipeline complet de fusion des 5 datasets.

    Args:
        config      : configuration globale (config.yaml)
        copy_images : si True, copie physique des images (consomme
                      beaucoup d'espace) ; si False (défaut), crée
                      des symlinks (léger, recommandé sur Colab/Drive)

    Returns:
        dict avec :
            - merged_df       : DataFrame complet fusionné
            - train_df / val_df / test_df
            - label2idx       : mapping classe → index
            - class_weights   : poids par classe (pour l'entraînement)
            - yolo_data_yaml  : chemin vers le data.yaml YOLO (DS5)
            - output_dir      : chemin racine du dataset fusionné
    """
    print("\n" + "=" * 60)
    print("  FUSION DES DATASETS — WheatAI-Merged")
    print("=" * 60)

    # Configuration
    merge_cfg   = config.get("merge", {})
    paths_cfg   = merge_cfg.get("paths", {})
    seed        = config["project"]["seed"]
    train_ratio = config["split"]["train_ratio"]
    val_ratio   = config["split"]["val_ratio"]
    output_dir  = Path(merge_cfg.get("output", "data/merged"))
    output_dir.mkdir(parents=True, exist_ok=True)

    roboflow_classes = merge_cfg.get(
        "roboflow_classes", ["Yellow_Rust", "Brown_Rust", "Stem_Rust"]
    )

    # ── 1. Collecte de chaque dataset ──
    print("\n[1/6] Collecte des images de chaque source...")
    all_records = []

    all_records.extend(_collect_ds1_wheat_plant_diseases(paths_cfg))
    all_records.extend(_collect_ds2_mendeley(paths_cfg))
    all_records.extend(_collect_ds3_yellow_rust_hayit(paths_cfg))
    all_records.extend(_collect_ds4_zenodo_brown(paths_cfg))
    ds5_result  = _collect_ds5_roboflow_yolo(paths_cfg)

    print(f"\n  Total brut : {len(all_records)} images (avant déduplication)")

    # ── 2. Déduplication ──
    print("\n[2/6] Déduplication...")
    all_records = _deduplicate(all_records)

    # ── 3. Structure de classification ──
    print("\n[3/6] Organisation par classe...")
    merged_df = _build_classification_structure(all_records, output_dir, copy_images)
    merged_df.rename(columns={"canonical": "label"}, inplace=True)

    # Encodage numérique
    label_names = sorted(merged_df["label"].unique().tolist())
    label2idx   = {name: i for i, name in enumerate(label_names)}
    idx2label   = {i: name for name, i in label2idx.items()}
    merged_df["label_idx"] = merged_df["label"].map(label2idx)

    # ── 4. Split stratifié ──
    print("\n[4/6] Split stratifié (70/15/15)...")
    merged_df_renamed = merged_df.rename(columns={"label": "canonical"})
    train_df, val_df, test_df = _stratified_split_df(
        merged_df_renamed, train_ratio, val_ratio, seed
    )
    train_df = train_df.rename(columns={"canonical": "label"})
    val_df   = val_df.rename(columns={"canonical": "label"})
    test_df  = test_df.rename(columns={"canonical": "label"})
    train_df["label_idx"] = train_df["label"].map(label2idx)
    val_df["label_idx"]   = val_df["label"].map(label2idx)
    test_df["label_idx"]  = test_df["label"].map(label2idx)

    # ── 5. Poids de classe ──
    print("\n[5/6] Calcul des poids de classe...")
    from sklearn.utils.class_weight import compute_class_weight

    cw_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_names)),
        y=train_df["label_idx"].values
    )
    class_weights = {i: float(w) for i, w in enumerate(cw_array)}
    print(f"  Ratio max/min : "
          f"{max(class_weights.values())/min(class_weights.values()):.2f}x")

    # ── 6. Sauvegarde ──
    print("\n[6/6] Sauvegarde...")
    proc_dir = output_dir / "processed"
    proc_dir.mkdir(exist_ok=True)

    train_df.to_csv(proc_dir / "merged_train.csv", index=False)
    val_df.to_csv(proc_dir   / "merged_val.csv",   index=False)
    test_df.to_csv(proc_dir  / "merged_test.csv",  index=False)

    metadata = {
        "dataset_name"  : "WheatAI-Merged",
        "sources"       : [
            "DS1_WheatPlantDiseases (Kaggle kushagra3204)",
            "DS2_Mendeley_Bangladesh (5gc7hwydwg)",
            "DS3_YellowRust19 (Kaggle tolgahayit)",
            "DS4_ZenodoWheatSmall (Brown 2023)",
            "DS5_RoboflowWheatRust (bboxes YOLO)",
        ],
        "num_classes"   : len(label_names),
        "label2idx"     : label2idx,
        "idx2label"     : {str(k): v for k, v in idx2label.items()},
        "class_weights" : {str(k): v for k, v in class_weights.items()},
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

    print(f"  train.csv   ({len(train_df)} lignes)")
    print(f"  val.csv     ({len(val_df)} lignes)")
    print(f"  test.csv    ({len(test_df)} lignes)")
    print(f"  metadata.json")

    # YOLO subset (DS5)
    yolo_yaml = ""
    if ds5_result["images"]:
        print(f"\n  Construction du sous-dataset YOLO (DS5 Roboflow)...")
        yolo_yaml = _build_yolo_subset(ds5_result, output_dir, roboflow_classes)
        print(f"  YOLO data.yaml : {yolo_yaml}")
    else:
        print(f"\n    DS5 Roboflow introuvable → sous-dataset YOLO ignoré")

    _print_merge_report(merged_df, train_df, val_df, test_df)

    print(f"\n   Dataset fusionné : {output_dir}")
    print("  Fusion terminée\n")

    return {
        "merged_df"    : merged_df,
        "train_df"     : train_df,
        "val_df"       : val_df,
        "test_df"      : test_df,
        "label2idx"    : label2idx,
        "idx2label"    : idx2label,
        "class_weights": class_weights,
        "yolo_data_yaml": yolo_yaml,
        "output_dir"   : str(output_dir),
    }