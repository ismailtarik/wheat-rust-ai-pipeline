"""
pipelines/obb_generation.py
------------------------------
Contribution 2 — Génération automatique d'annotations OBB (Oriented
Bounding Box) à partir de bounding boxes axis-aligned existantes.

Motivation scientifique :
    Les lésions de rouille striée et autres maladies du blé présentent
    souvent une géométrie allongée et orientée qui n'est pas efficacement
    capturée par une bbox axis-aligned (la boîte doit alors être bien plus
    grande que la lésion réelle pour la contenir entièrement). Aucune
    annotation OBB réelle n'étant disponible publiquement pour ce domaine,
    cette méthode dérive des OBB à partir des bboxes existantes par
    segmentation colorimétrique + ajustement de rectangle à aire minimale,
    sans nécessiter de ré-annotation manuelle.

Méthode (par bbox axis-aligned d'entrée) :
    1. Extraction de la région d'intérêt (crop) avec une marge de sécurité
    2. Segmentation colorimétrique : distance de teinte (HSV) par rapport
       à une référence "tissu sain" (vert), suivie d'un seuillage Otsu sur
       la carte de distance — isole les pixels visuellement atypiques
       (couleurs de maladie : brun, jaune, noir, orangé...)
    3. Nettoyage morphologique (ouverture/fermeture) pour éliminer le bruit
    4. Extraction du plus grand contour connecté
    5. Ajustement d'un rectangle orienté à aire minimale (cv2.minAreaRect)
       autour de ce contour
    6. Conversion en coordonnées normalisées (image complète) + validation
       géométrique (ratio d'aire, aspect ratio)
    7. En cas d'échec de segmentation (contour trop petit/absent), repli
       sur la bbox axis-aligned d'origine convertie en OBB à 0° de rotation
       (garantit qu'aucune annotation n'est perdue, tout en traçant le taux
       de repli pour transparence méthodologique dans l'article)

Usage :
    from pipelines.obb_generation import build_obb_dataset
    result = build_obb_dataset(yolo_root="data/merged/yolo", output_root="data/merged/yolo_obb")
"""

import json
import shutil
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Segmentation colorimétrique
# ─────────────────────────────────────────────────────────────

# Teinte HSV de référence pour un tissu de blé sain (vert), en degrés OpenCV
# (H dans [0,180]). Les pixels de maladie (brun/jaune/noir/orangé) s'en
# écartent significativement.
HEALTHY_HUE_REF = 45   # vert-jaune (OpenCV H range 0-180)
HEALTHY_HUE_TOLERANCE = 25


def _segment_disease_region(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Segmente les pixels visuellement atypiques (probable maladie) dans un
    crop BGR, par distance de teinte à la référence "sain" + seuillage Otsu.

    Returns:
        Masque binaire (uint8, 0/255) de même taille que crop_bgr
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    # Distance angulaire circulaire à la teinte de référence (0-180 -> circulaire)
    diff = np.abs(hue - HEALTHY_HUE_REF)
    diff = np.minimum(diff, 180 - diff)

    # Pondération par la saturation : un pixel peu saturé (quasi gris/blanc,
    # ex: reflet, fond) ne doit pas être classé "maladie" même s'il a une
    # teinte éloignée du vert de référence.
    distance_map = diff * (sat / 255.0)

    # Seuillage Otsu sur la carte de distance (normalisée en uint8)
    dist_norm = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _largest_contour(mask: np.ndarray, min_area_px: int = 20) -> "np.ndarray | None":
    """Retourne le plus grand contour du masque, ou None si aucun n'est valide."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None
    return largest


# ─────────────────────────────────────────────────────────────
# Conversion bbox → OBB
# ─────────────────────────────────────────────────────────────

def _bbox_to_pixel_coords(box: tuple, img_w: int, img_h: int,
                           margin_ratio: float = 0.15) -> tuple:
    """
    Convertit une bbox YOLO normalisée (xc, yc, w, h) en coordonnées pixel
    (x0, y0, x1, y1), avec une marge de sécurité pour ne pas couper la
    lésion si la bbox d'origine est légèrement trop serrée.
    """
    _, xc, yc, bw, bh = box
    xc, yc, bw, bh = xc * img_w, yc * img_h, bw * img_w, bh * img_h

    bw *= (1 + margin_ratio)
    bh *= (1 + margin_ratio)

    x0 = max(0, int(xc - bw / 2))
    y0 = max(0, int(yc - bh / 2))
    x1 = min(img_w, int(xc + bw / 2))
    y1 = min(img_h, int(yc + bh / 2))
    return x0, y0, x1, y1


def _axis_aligned_fallback_obb(box: tuple, img_w: int, img_h: int) -> list:
    """
    Convertit une bbox axis-aligned en OBB à 0° de rotation (repli utilisé
    quand la segmentation échoue) — les 4 coins du rectangle non-rotaté,
    normalisés à l'image complète.
    """
    _, xc, yc, bw, bh = box
    x0, y0 = xc - bw / 2, yc - bh / 2
    x1, y1 = xc + bw / 2, yc + bh / 2
    # Ordre des coins : haut-gauche, haut-droit, bas-droit, bas-gauche
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _validate_obb(points_norm: list, min_area_ratio: float = 0.005,
                   max_area_ratio: float = 1.0, max_aspect_ratio: float = 15.0) -> bool:
    """
    Valide la géométrie d'un OBB : aire raisonnable (ni dégénérée ni
    aberrante) et aspect ratio borné (évite les artefacts de segmentation
    donnant des rectangles quasi-linéaires).
    """
    pts = np.array(points_norm).reshape(4, 2)
    # Aire via la formule du polygone (shoelace)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    if area < min_area_ratio or area > max_area_ratio:
        return False

    side1 = np.linalg.norm(pts[1] - pts[0])
    side2 = np.linalg.norm(pts[2] - pts[1])
    if min(side1, side2) < 1e-6:
        return False
    aspect = max(side1, side2) / min(side1, side2)
    if aspect > max_aspect_ratio:
        return False

    return True


def _generate_obb_for_box(image_bgr: np.ndarray, box: tuple) -> tuple:
    """
    Génère un OBB pour une bbox axis-aligned donnée.

    Returns:
        (points_norm, is_fallback) — points_norm : liste de 8 flottants
        normalisés [x1,y1,x2,y2,x3,y3,x4,y4] (coordonnées image complète) ;
        is_fallback : True si la segmentation a échoué et qu'on retombe
        sur la bbox axis-aligned.
    """
    img_h, img_w = image_bgr.shape[:2]
    x0, y0, x1, y1 = _bbox_to_pixel_coords(box, img_w, img_h)

    if x1 <= x0 or y1 <= y0:
        return _axis_aligned_fallback_obb(box, img_w, img_h), True

    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return _axis_aligned_fallback_obb(box, img_w, img_h), True

    mask = _segment_disease_region(crop)
    contour = _largest_contour(mask, min_area_px=max(20, crop.shape[0] * crop.shape[1] * 0.02))

    if contour is None:
        return _axis_aligned_fallback_obb(box, img_w, img_h), True

    # Rectangle orienté à aire minimale autour du contour (coordonnées crop)
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect)  # 4 points (x,y) dans le repère du crop

    # Repositionnement dans le repère image complète + normalisation
    box_pts[:, 0] += x0
    box_pts[:, 1] += y0
    box_pts[:, 0] /= img_w
    box_pts[:, 1] /= img_h

    points_norm = box_pts.flatten().tolist()

    if not _validate_obb(points_norm):
        return _axis_aligned_fallback_obb(box, img_w, img_h), True

    return points_norm, False


# ─────────────────────────────────────────────────────────────
# Orchestrateur — conversion d'un dataset YOLO complet vers OBB
# ─────────────────────────────────────────────────────────────

def _process_split(yolo_root: Path, output_root: Path, split: str) -> dict:
    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split
    out_img_dir = output_root / "images" / split
    out_lbl_dir = output_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_total_boxes = 0
    n_fallback = 0
    n_images_processed = 0

    if not img_dir.exists():
        return {"total_boxes": 0, "fallback_boxes": 0, "images_processed": 0}

    for img_path in sorted(img_dir.iterdir()):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_idx = int(float(parts[0]))
                    coords = [float(p) for p in parts[1:5]]
                    boxes.append((cls_idx, *coords))

        if not boxes:
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        obb_lines = []
        for box in boxes:
            points_norm, is_fallback = _generate_obb_for_box(image_bgr, box)
            points_norm = [max(0.0, min(1.0, p)) for p in points_norm]
            line = f"{box[0]} " + " ".join(f"{p:.6f}" for p in points_norm)
            obb_lines.append(line)
            n_total_boxes += 1
            if is_fallback:
                n_fallback += 1

        # Écriture du label OBB + lien vers l'image (pas de copie physique)
        dst_lbl = out_lbl_dir / lbl_path.name
        dst_lbl.write_text("\n".join(obb_lines) + "\n")

        dst_img = out_img_dir / img_path.name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except Exception:
                shutil.copy2(img_path, dst_img)

        n_images_processed += 1

    return {
        "total_boxes": n_total_boxes,
        "fallback_boxes": n_fallback,
        "images_processed": n_images_processed,
    }


def build_obb_dataset(yolo_root: str, output_root: str,
                       class_names: list = None) -> dict:
    """
    Convertit un dataset YOLO (bbox axis-aligned) en dataset YOLO-OBB par
    segmentation colorimétrique + ajustement de rectangle orienté.

    Args:
        yolo_root    : racine du dataset YOLO source (images/labels bbox)
        output_root  : racine du nouveau dataset OBB généré
        class_names  : liste des noms de classes (pour le data.yaml généré) ;
                       si None, tente de les lire depuis yolo_root/data.yaml

    Returns:
        dict avec les statistiques de génération (nb OBB réels vs repli,
        par split) et le chemin du data.yaml généré
    """
    if not HAS_CV2:
        raise ImportError(
            "❌ opencv-python requis pour la génération OBB. "
            "pip install opencv-python-headless"
        )

    print("\n" + "=" * 60)
    print("  GÉNÉRATION AUTOMATIQUE D'ANNOTATIONS OBB")
    print("  (segmentation colorimétrique + rectangle orienté minimal)")
    print("=" * 60)

    yolo_root = Path(yolo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Récupération des noms de classes depuis le data.yaml source
    if class_names is None:
        import yaml as yaml_lib
        src_yaml = yolo_root / "data.yaml"
        if src_yaml.exists():
            with open(src_yaml) as f:
                content = yaml_lib.safe_load(f)
            class_names = content.get("names", [])
        else:
            class_names = []

    stats = {}
    for split in ["train", "val", "test"]:
        print(f"\n  Traitement du split '{split}'...")
        split_stats = _process_split(yolo_root, output_root, split)
        stats[split] = split_stats

        if split_stats["total_boxes"] > 0:
            pct_real = 100 * (1 - split_stats["fallback_boxes"] / split_stats["total_boxes"])
            print(f"    {split_stats['images_processed']} images | "
                  f"{split_stats['total_boxes']} boîtes | "
                  f"{pct_real:.1f}% OBB réellement orientés "
                  f"({split_stats['fallback_boxes']} replis axis-aligned)")
        else:
            print(f"    Aucune image/label trouvé pour ce split")

    # data.yaml pour le dataset OBB
    import yaml as yaml_lib
    data_yaml = {
        "path" : str(output_root.resolve()),
        "train": "images/train",
        "val"  : "images/val",
        "test" : "images/test",
        "nc"   : len(class_names),
        "names": class_names,
    }
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml_lib.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    total_boxes = sum(s["total_boxes"] for s in stats.values())
    total_fallback = sum(s["fallback_boxes"] for s in stats.values())
    pct_real_overall = (100 * (1 - total_fallback / total_boxes)) if total_boxes else 0

    summary = {
        "yolo_root": str(yolo_root),
        "output_root": str(output_root),
        "data_yaml": str(yaml_path),
        "per_split": stats,
        "total_boxes": total_boxes,
        "total_fallback_boxes": total_fallback,
        "pct_true_obb": round(pct_real_overall, 2),
    }

    with open(output_root / "obb_generation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✅ data.yaml OBB : {yaml_path}")
    print(f"  ✅ Rapport de génération : {output_root / 'obb_generation_report.json'}")
    print(f"\n  Résumé global : {total_boxes} boîtes traitées, "
          f"{pct_real_overall:.1f}% avec orientation réellement détectée "
          f"({total_fallback} replis axis-aligned sur segmentation infructueuse)")
    print("=" * 60 + "\n")

    return summary