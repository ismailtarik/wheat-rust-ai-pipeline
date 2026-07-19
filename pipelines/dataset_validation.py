"""
pipelines/dataset_validation.py
----------------------------------
Phase 3 — Validation complète du dataset (EDA approfondie pour article).

Génère automatiquement, pour le dataset fusionné (ou original) :
    1.  Total images (Healthy / Diseased / Detection / Classification)
    2.  Distribution des classes (histogramme)
    3.  Résolutions (histogrammes largeur/hauteur)
    4.  Distribution des formats de fichier (jpg/png/...)
    5.  Aspect ratio (histogramme)
    6.  Mode couleur (RGB / RGBA / Grayscale)
    7.  Tableau images corrompues / labels manquants / fichiers invalides
    8.  Doublons (perceptual hash — pHash + aHash)
    9.  Statistiques d'annotation YOLO (bbox) + OBB si détecté
    10. Nombre d'objets par image (histogramme)
    11. Heatmap des bounding boxes
    12. Visualisation aléatoire (1 exemple par classe)
    13. Exemples d'annotations (original → YOLO boxes → OBB si dispo)
    14. Rapport PDF final assemblant toutes les figures

Toutes les opérations coûteuses (résolution, format, couleur, doublons)
sont faites sur un ÉCHANTILLON aléatoire configurable (par défaut 3000
images) pour rester praticable sur Colab — la distribution de classes
(point 2), elle, utilise le dataset COMPLET (simple comptage de lignes,
pas d'ouverture d'image).

Usage :
    from pipelines.dataset_validation import run_dataset_validation
    run_dataset_validation(config, data_source="merged")
"""

import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Polygon

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


CLASS_COLORS_FALLBACK = [
    "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E", "#16A085",
    "#D35400", "#7F8C8D", "#2980B9", "#8E44AD", "#27AE60", "#C0392B",
    "#F39C12", "#95A5A6", "#E74C3C", "#2ECC71", "#F1C40F",
]


# ─────────────────────────────────────────────────────────────
# Chargement du dataset
# ─────────────────────────────────────────────────────────────

def _load_classification_df(config: dict, data_source: str) -> pd.DataFrame:
    """Charge le DataFrame complet (train+val+test) avec colonnes filepath/label/split."""
    if data_source == "merged":
        proc_dir = Path(config["merge"]["output"]) / "processed"
        files = {"train": "merged_train.csv", "val": "merged_val.csv", "test": "merged_test.csv"}
    else:
        proc_dir = Path(config["paths"]["processed"])
        files = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

    dfs = []
    for split, fname in files.items():
        path = proc_dir / fname
        if path.exists():
            d = pd.read_csv(path)
            d["split"] = split
            dfs.append(d)

    if not dfs:
        raise FileNotFoundError(
            f"❌ Aucun CSV trouvé dans {proc_dir}. "
            f"Lance d'abord la Phase 1 ou la fusion (--phase merge)."
        )

    return pd.concat(dfs, ignore_index=True)


def _get_yolo_root(config: dict, data_source: str) -> "Path | None":
    """Retourne le chemin racine du dataset YOLO (détection), si disponible."""
    if data_source == "merged":
        yolo_root = Path(config["merge"]["output"]) / "yolo"
    else:
        yolo_root = Path(config["phase2"]["paths"].get("yolo_dataset", ""))

    return yolo_root if yolo_root.exists() else None


def _get_class_colors(label_names: list) -> dict:
    colors = {}
    for i, label in enumerate(sorted(label_names)):
        colors[label] = CLASS_COLORS_FALLBACK[i % len(CLASS_COLORS_FALLBACK)]
    return colors


# ─────────────────────────────────────────────────────────────
# 1. Total images
# ─────────────────────────────────────────────────────────────

def _stat_total_images(df: pd.DataFrame, yolo_root) -> dict:
    total = len(df)
    healthy = int((df["label"].str.lower() == "healthy").sum())
    diseased = total - healthy

    n_detection = 0
    if yolo_root:
        for split in ["train", "val", "test"]:
            img_dir = yolo_root / "images" / split
            if img_dir.exists():
                n_detection += len(list(img_dir.iterdir()))

    stats = {
        "total_images"      : total,
        "healthy_images"    : healthy,
        "diseased_images"   : diseased,
        "classification_images": total,
        "detection_images"  : n_detection,
    }
    return stats


# ─────────────────────────────────────────────────────────────
# 2. Distribution des classes
# ─────────────────────────────────────────────────────────────

def _stat_class_distribution(df: pd.DataFrame, save_path: Path) -> dict:
    counts = df["label"].value_counts().sort_values()
    colors_map = _get_class_colors(counts.index.tolist())
    colors = [colors_map[c] for c in counts.index]

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.35)))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(val + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
               str(val), va="center", fontsize=9)
    ax.set_xlabel("Nombre d'images")
    ax.set_title("Distribution des classes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return counts.to_dict()


# ─────────────────────────────────────────────────────────────
# 3-6. Résolution / format / aspect ratio / mode couleur (échantillon)
# ─────────────────────────────────────────────────────────────

def _scan_image_properties(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Ouvre un échantillon d'images pour extraire largeur, hauteur, format,
    mode couleur. Retourne un DataFrame avec ces propriétés (+ filepath).
    """
    sample = df.sample(min(sample_size, len(df)), random_state=seed)
    rows = []
    for fp in sample["filepath"]:
        try:
            with Image.open(fp) as img:
                w, h = img.size
                mode = img.mode
                fmt = Path(fp).suffix.lower().lstrip(".")
                rows.append({"filepath": fp, "width": w, "height": h,
                            "mode": mode, "format": fmt})
        except Exception:
            continue
    return pd.DataFrame(rows)


def _stat_resolution(props_df: pd.DataFrame, save_dir: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(props_df["width"], bins=30, color="#3498DB", edgecolor="white")
    ax.set_xlabel("Largeur (px)")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des largeurs d'image", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "03a_width_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(props_df["height"], bins=30, color="#E67E22", edgecolor="white")
    ax.set_xlabel("Hauteur (px)")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des hauteurs d'image", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "03b_height_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "width_min" : int(props_df["width"].min()),
        "width_max" : int(props_df["width"].max()),
        "width_mean": round(float(props_df["width"].mean()), 1),
        "height_min" : int(props_df["height"].min()),
        "height_max" : int(props_df["height"].max()),
        "height_mean": round(float(props_df["height"].mean()), 1),
    }


def _stat_format(props_df: pd.DataFrame, save_path: Path) -> dict:
    counts = props_df["format"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index, counts.values, color="#9B59B6", edgecolor="white")
    ax.set_xlabel("Format")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des formats de fichier", fontsize=12, fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return counts.to_dict()


def _classify_aspect_ratio(w: int, h: int) -> str:
    """Classe un ratio largeur/hauteur dans la catégorie standard la plus proche."""
    ratio = w / h if h else 0
    common = {"1:1": 1.0, "4:3": 4/3, "3:2": 3/2, "16:9": 16/9, "3:4": 3/4, "9:16": 9/16}
    closest = min(common.items(), key=lambda kv: abs(kv[1] - ratio))
    return closest[0]


def _stat_aspect_ratio(props_df: pd.DataFrame, save_path: Path) -> dict:
    ratios = [_classify_aspect_ratio(w, h)
              for w, h in zip(props_df["width"], props_df["height"])]
    counts = Counter(ratios)
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, values, color="#16A085", edgecolor="white")
    ax.set_xlabel("Aspect ratio")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des aspect ratios", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return dict(counts)


def _stat_color_mode(props_df: pd.DataFrame, save_path: Path) -> dict:
    mode_map = {"RGB": "RGB", "RGBA": "RGBA", "L": "Grayscale", "P": "Palette (P)"}
    modes = props_df["mode"].map(lambda m: mode_map.get(m, m))
    counts = modes.value_counts()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
          colors=CLASS_COLORS_FALLBACK[:len(counts)],
          wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Mode couleur des images", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return counts.to_dict()


# ─────────────────────────────────────────────────────────────
# 7. Images corrompues / labels manquants / fichiers invalides
# ─────────────────────────────────────────────────────────────

def _stat_corrupted(df: pd.DataFrame, yolo_root, sample_size: int, seed: int) -> dict:
    sample = df.sample(min(sample_size, len(df)), random_state=seed)
    n_corrupted = 0
    n_invalid = 0

    for fp in sample["filepath"]:
        path = Path(fp)
        if not path.exists():
            n_invalid += 1
            continue
        if path.stat().st_size == 0:
            n_invalid += 1
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            n_corrupted += 1

    n_missing_labels = 0
    if yolo_root:
        for split in ["train", "val", "test"]:
            img_dir = yolo_root / "images" / split
            lbl_dir = yolo_root / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    n_missing_labels += 1

    return {
        "sample_size_checked": len(sample),
        "corrupted_images"   : n_corrupted,
        "missing_labels"     : n_missing_labels,
        "invalid_files"      : n_invalid,
    }


# ─────────────────────────────────────────────────────────────
# 8. Doublons (perceptual hash)
# ─────────────────────────────────────────────────────────────

def _stat_duplicates(df: pd.DataFrame, sample_size: int, seed: int,
                      remove: bool = False) -> dict:
    if not HAS_IMAGEHASH:
        return {
            "available": False,
            "note": "Bibliothèque 'imagehash' non installée — "
                     "pip install imagehash pour activer cette analyse.",
        }

    sample = df.sample(min(sample_size, len(df)), random_state=seed)
    hashes = {}
    duplicate_groups = defaultdict(list)

    for fp in sample["filepath"]:
        try:
            with Image.open(fp) as img:
                h_phash = imagehash.phash(img)
        except Exception:
            continue
        duplicate_groups[str(h_phash)].append(fp)

    n_duplicates_found = sum(len(v) - 1 for v in duplicate_groups.values() if len(v) > 1)
    n_removed = 0

    if remove:
        for group in duplicate_groups.values():
            if len(group) > 1:
                for dup_fp in group[1:]:
                    try:
                        Path(dup_fp).unlink()
                        n_removed += 1
                    except Exception:
                        pass

    return {
        "available"          : True,
        "sample_size_checked": len(sample),
        "duplicates_found"   : n_duplicates_found,
        "duplicates_removed" : n_removed,
        "remaining_images"   : len(sample) - n_removed,
        "method"             : "perceptual hash (pHash, 8x8)",
    }


# ─────────────────────────────────────────────────────────────
# 9-11. Statistiques d'annotation YOLO / OBB / heatmap
# ─────────────────────────────────────────────────────────────

def _parse_label_file(path: Path) -> tuple:
    """Retourne (mode, boxes) où mode = 'bbox' (5 val/ligne) ou 'obb' (9 val/ligne)."""
    boxes = []
    mode = None
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 5:
                    mode = mode or "bbox"
                    boxes.append([float(p) for p in parts])
                elif len(parts) == 9:
                    mode = mode or "obb"
                    boxes.append([float(p) for p in parts])
    except Exception:
        pass
    return mode, boxes


def _stat_annotation_yolo(yolo_root: Path, save_dir: Path, idx2label: dict = None) -> dict:
    all_boxes_bbox = []   # (class_idx, x, y, w, h) normalisé
    all_boxes_obb  = []   # (class_idx, x1,y1,...,x4,y4) normalisé
    objects_per_image = []

    for split in ["train", "val", "test"]:
        lbl_dir = yolo_root / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_path in lbl_dir.glob("*.txt"):
            mode, boxes = _parse_label_file(lbl_path)
            objects_per_image.append(len(boxes))
            if mode == "bbox":
                all_boxes_bbox.extend(boxes)
            elif mode == "obb":
                all_boxes_obb.extend(boxes)

    stats = {"bbox": {}, "obb": {}, "objects_per_image": {}}

    # --- BBOX stats ---
    if all_boxes_bbox:
        areas = [b[3] * b[4] for b in all_boxes_bbox]  # w*h normalisé
        per_class = Counter(int(b[0]) for b in all_boxes_bbox)
        per_class_named = ({idx2label.get(str(k), idx2label.get(k, str(k))): v
                            for k, v in per_class.items()}
                           if idx2label else {str(k): v for k, v in per_class.items()})

        stats["bbox"] = {
            "total_boxes"      : len(all_boxes_bbox),
            "avg_area_normalized": round(float(np.mean(areas)), 5),
            "min_area_normalized": round(float(np.min(areas)), 5),
            "max_area_normalized": round(float(np.max(areas)), 5),
            "boxes_per_class"  : per_class_named,
        }

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(areas, bins=30, color="#E74C3C", edgecolor="white")
        ax.set_xlabel("Aire de la bbox (normalisée, w×h)")
        ax.set_ylabel("Nombre de boîtes")
        ax.set_title("Distribution des aires de bounding boxes", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "09_bbox_area_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- OBB stats ---
    if all_boxes_obb:
        def poly_area(coords):
            xs = coords[0::2]
            ys = coords[1::2]
            return 0.5 * abs(sum(xs[i]*ys[(i+1) % 4] - xs[(i+1) % 4]*ys[i] for i in range(4)))

        def poly_angle(coords):
            xs = coords[0::2]
            ys = coords[1::2]
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            return float(np.degrees(np.arctan2(dy, dx)))

        areas_obb = [poly_area(b[1:]) for b in all_boxes_obb]
        angles = [poly_angle(b[1:]) for b in all_boxes_obb]

        stats["obb"] = {
            "total_polygons"   : len(all_boxes_obb),
            "avg_area_normalized": round(float(np.mean(areas_obb)), 5),
            "avg_orientation_deg": round(float(np.mean(angles)), 2),
            "vertices_per_polygon": 4,
        }
    else:
        stats["obb"] = {"note": "Aucune annotation OBB détectée dans ce dataset."}

    # --- Objects per image ---
    if objects_per_image:
        stats["objects_per_image"] = {
            "mean": round(float(np.mean(objects_per_image)), 2),
            "min" : int(np.min(objects_per_image)),
            "max" : int(np.max(objects_per_image)),
        }

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(objects_per_image, bins=range(0, max(objects_per_image) + 2),
               color="#2ECC71", edgecolor="white")
        ax.set_xlabel("Nombre d'objets par image")
        ax.set_ylabel("Nombre d'images")
        ax.set_title("Distribution du nombre d'objets par image",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "10_objects_per_image.png", dpi=150, bbox_inches="tight")
        plt.close()

    return stats, all_boxes_bbox


def _stat_bbox_heatmap(all_boxes_bbox: list, save_path: Path, grid_size: int = 50) -> None:
    """Génère une heatmap montrant où les bounding boxes apparaissent le plus souvent."""
    if not all_boxes_bbox:
        return

    heatmap = np.zeros((grid_size, grid_size))
    for box in all_boxes_bbox:
        _, x, y, w, h = box
        gx = min(int(x * grid_size), grid_size - 1)
        gy = min(int(y * grid_size), grid_size - 1)
        heatmap[gy, gx] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap="hot", interpolation="bilinear")
    ax.set_title("Heatmap des centres de bounding boxes", fontsize=12, fontweight="bold")
    ax.set_xlabel("Position X (normalisée)")
    ax.set_ylabel("Position Y (normalisée)")
    plt.colorbar(im, ax=ax, label="Densité d'annotations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 12. Visualisation aléatoire par classe
# ─────────────────────────────────────────────────────────────

def _fig_random_samples_per_class(df: pd.DataFrame, save_path: Path, seed: int) -> None:
    label_names = sorted(df["label"].unique())
    colors_map = _get_class_colors(label_names)

    n = len(label_names)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, label in enumerate(label_names):
        ax = axes_flat[i]
        sample_fp = df[df["label"] == label]["filepath"].sample(1, random_state=seed).iloc[0]
        try:
            with Image.open(sample_fp) as img:
                ax.imshow(img.convert("RGB"))
        except Exception:
            pass
        ax.set_title(label, fontsize=9, fontweight="bold", color=colors_map[label])
        ax.axis("off")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Exemple aléatoire par classe", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 13. Exemples d'annotations (original → YOLO boxes → OBB)
# ─────────────────────────────────────────────────────────────

def _fig_annotation_examples(yolo_root: Path, save_path: Path,
                              idx2label: dict = None, n_examples: int = 3) -> None:
    if not yolo_root:
        return

    examples = []
    for split in ["train", "val", "test"]:
        lbl_dir = yolo_root / "labels" / split
        img_dir = yolo_root / "images" / split
        if not lbl_dir.exists():
            continue
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            mode, boxes = _parse_label_file(lbl_path)
            if boxes:
                img_candidates = list(img_dir.glob(lbl_path.stem + ".*"))
                if img_candidates:
                    examples.append((img_candidates[0], boxes, mode))
            if len(examples) >= n_examples:
                break
        if len(examples) >= n_examples:
            break

    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 2, figsize=(8, len(examples) * 4))
    if len(examples) == 1:
        axes = axes.reshape(1, -1)

    for row, (img_path, boxes, mode) in enumerate(examples):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        w, h = img.size

        axes[row][0].imshow(img)
        axes[row][0].set_title("Original", fontsize=10)
        axes[row][0].axis("off")

        axes[row][1].imshow(img)
        for box in boxes:
            cls_idx = int(box[0])
            label = (idx2label.get(str(cls_idx), idx2label.get(cls_idx, str(cls_idx)))
                    if idx2label else str(cls_idx))
            if mode == "bbox":
                _, xc, yc, bw, bh = box
                x0 = (xc - bw / 2) * w
                y0 = (yc - bh / 2) * h
                rect = Rectangle((x0, y0), bw * w, bh * h,
                                linewidth=2, edgecolor="#E74C3C", facecolor="none")
                axes[row][1].add_patch(rect)
                axes[row][1].text(x0, y0 - 3, label, color="#E74C3C", fontsize=8,
                                  fontweight="bold")
            elif mode == "obb":
                coords = box[1:]
                pts = [(coords[i] * w, coords[i+1] * h) for i in range(0, 8, 2)]
                poly = Polygon(pts, closed=True, linewidth=2,
                               edgecolor="#2ECC71", facecolor="none")
                axes[row][1].add_patch(poly)
        axes[row][1].set_title(f"Annotations ({mode})", fontsize=10)
        axes[row][1].axis("off")

    fig.suptitle("Exemples d'annotations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 14. Rapport PDF final
# ─────────────────────────────────────────────────────────────

def _build_pdf_report(reports_dir: Path, all_stats: dict, pdf_path: Path) -> None:
    """Assemble toutes les figures PNG générées + une page de résumé texte en un seul PDF."""
    with PdfPages(pdf_path) as pdf:
        # Page de titre / résumé
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
        ax.axis("off")
        lines = [
            "Dataset Validation Report",
            "WheatAI-Merged — Wheat Disease Dataset",
            f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "── Résumé ──",
        ]
        total = all_stats.get("total_images", {})
        for k, v in total.items():
            lines.append(f"{k.replace('_', ' ').title()} : {v}")

        lines.append("")
        lines.append("── Résolution ──")
        for k, v in all_stats.get("resolution", {}).items():
            lines.append(f"{k} : {v}")

        lines.append("")
        lines.append("── Images corrompues / manquantes ──")
        for k, v in all_stats.get("corrupted", {}).items():
            lines.append(f"{k} : {v}")

        lines.append("")
        lines.append("── Doublons ──")
        for k, v in all_stats.get("duplicates", {}).items():
            lines.append(f"{k} : {v}")

        ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10,
               family="monospace", transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close()

        # Insertion de chaque figure PNG générée
        png_files = sorted(reports_dir.glob("*.png"))
        for png_path in png_files:
            try:
                img = plt.imread(png_path)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(png_path.stem.replace("_", " "), fontsize=11)
                pdf.savefig(fig)
                plt.close()
            except Exception:
                continue


# ─────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────

def run_dataset_validation(config: dict, data_source: str = "merged",
                            sample_size: int = 3000,
                            remove_duplicates: bool = False) -> dict:
    """
    Pipeline complet de validation du dataset (Phase 3 EDA approfondie).

    Args:
        config            : configuration globale
        data_source       : "merged" ou "original"
        sample_size       : nombre d'images échantillonnées pour les
                            opérations coûteuses (résolution, format,
                            couleur, doublons). La distribution de
                            classes utilise toujours le dataset complet.
        remove_duplicates : si True, supprime physiquement les doublons
                            détectés (pHash). Par défaut False (non
                            destructif — rapport uniquement).

    Returns:
        dict avec toutes les statistiques calculées
    """
    print("\n" + "=" * 65)
    print("  PHASE 3 — Validation Complète du Dataset")
    print("=" * 65)

    seed = config["project"]["seed"]
    reports_dir = Path(config["paths"]["reports"]) / "validation"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Source        : {data_source}")
    print(f"  Échantillon   : {sample_size} images (opérations coûteuses)")
    print(f"  Dossier sortie: {reports_dir}")

    df = _load_classification_df(config, data_source)
    yolo_root = _get_yolo_root(config, data_source)

    idx2label = None
    meta_path = (Path(config["merge"]["output"]) / "processed" / "merged_metadata.json"
                if data_source == "merged"
                else Path(config["paths"]["metadata_json"]))
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        idx2label = meta.get("idx2label", {})

    all_stats = {}

    print("\n[1/14] Total images...")
    all_stats["total_images"] = _stat_total_images(df, yolo_root)
    print(f"  {all_stats['total_images']}")

    print("\n[2/14] Distribution des classes...")
    all_stats["class_distribution"] = _stat_class_distribution(
        df, reports_dir / "02_class_distribution.png"
    )

    print("\n[3-6/14] Propriétés d'image (résolution, format, aspect ratio, couleur)...")
    props_df = _scan_image_properties(df, sample_size, seed)
    if len(props_df):
        all_stats["resolution"]   = _stat_resolution(props_df, reports_dir)
        all_stats["format"]       = _stat_format(props_df, reports_dir / "04_format_distribution.png")
        all_stats["aspect_ratio"] = _stat_aspect_ratio(props_df, reports_dir / "05_aspect_ratio.png")
        all_stats["color_mode"]   = _stat_color_mode(props_df, reports_dir / "06_color_mode.png")
    else:
        print("  ⚠️  Aucune image lisible dans l'échantillon")
        all_stats["resolution"] = all_stats["format"] = {}
        all_stats["aspect_ratio"] = all_stats["color_mode"] = {}

    print("\n[7/14] Images corrompues / labels manquants...")
    all_stats["corrupted"] = _stat_corrupted(df, yolo_root, sample_size, seed)
    print(f"  {all_stats['corrupted']}")

    print("\n[8/14] Détection de doublons (perceptual hash)...")
    all_stats["duplicates"] = _stat_duplicates(df, sample_size, seed, remove=remove_duplicates)
    print(f"  {all_stats['duplicates']}")

    if yolo_root:
        print("\n[9-11/14] Statistiques d'annotation YOLO/OBB + heatmap...")
        annotation_stats, all_boxes_bbox = _stat_annotation_yolo(
            yolo_root, reports_dir, idx2label
        )
        all_stats["annotation"] = annotation_stats
        _stat_bbox_heatmap(all_boxes_bbox, reports_dir / "11_bbox_heatmap.png")
    else:
        print("\n[9-11/14] Pas de dataset YOLO détecté — sections annotation ignorées")
        all_stats["annotation"] = {"note": "Aucun dataset de détection trouvé"}

    print("\n[12/14] Visualisation aléatoire par classe...")
    _fig_random_samples_per_class(df, reports_dir / "12_random_samples_per_class.png", seed)

    print("\n[13/14] Exemples d'annotations...")
    _fig_annotation_examples(yolo_root, reports_dir / "13_annotation_examples.png", idx2label)

    print("\n[14/14] Génération du rapport PDF final...")
    pdf_path = reports_dir / "dataset_report.pdf"
    _build_pdf_report(reports_dir, all_stats, pdf_path)
    print(f"  ✅ {pdf_path}")

    stats_json_path = reports_dir / "validation_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  📁 Tous les résultats sauvegardés dans : {reports_dir}")
    print("  ✅ Validation du dataset terminée\n")

    return all_stats