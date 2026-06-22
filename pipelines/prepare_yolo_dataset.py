"""
pipelines/prepare_yolo_dataset.py
------------------------------------
Phase 2 (Détection) — Étape 1 : Conversion du dataset de classification
(dossiers de classes, sans bounding boxes) vers le format YOLO.

Contexte (cf. discussion projet) : le dataset Wheat Plant Diseases ne
fournit que des labels de classification (1 dossier = 1 classe), sans
annotations de localisation. En l'absence de bounding boxes réelles, on
utilise le mode 'full_image' : chaque image reçoit une unique bbox
couvrant (quasi) toute l'image, avec une légère marge retirée des bords
(bbox_margin). Cela permet de faire tourner le pipeline YOLO complet
(format de données, entraînement, métriques mAP) dès maintenant ; si un
dataset annoté avec de vraies bounding boxes est obtenu plus tard, seules
les données changent — pas le code d'entraînement.

Structure générée (format YOLO standard) :
    data/processed/yolo_dataset/
    ├── images/
    │   ├── train/*.jpg (copiés ou liens symboliques)
    │   ├── val/*.jpg
    │   └── test/*.jpg
    ├── labels/
    │   ├── train/*.txt   (1 ligne par image : "class_idx xc yc w h", normalisé [0,1])
    │   ├── val/*.txt
    │   └── test/*.txt
    └── data.yaml          (chemins + noms de classes, attendu par Ultralytics)

Usage :
    from pipelines.prepare_yolo_dataset import build_yolo_dataset
    yolo_data_yaml = build_yolo_dataset(split_result, config)
"""

import shutil
import yaml as yaml_lib

from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback silencieux si tqdm n'est pas installé."""
        return iterable


def _make_full_image_label(class_idx: int, margin: float) -> str:
    """
    Construit une ligne d'annotation YOLO pour une bbox couvrant (quasi)
    toute l'image, avec une marge retirée des bords.

    Format YOLO : "class_idx x_center y_center width height" — toutes
    les valeurs normalisées dans [0, 1].

    Args:
        class_idx : index de classe (entier, 0-based)
        margin    : fraction retirée de chaque bord (ex: 0.05 = 5%)
    """
    width  = 1.0 - 2 * margin
    height = 1.0 - 2 * margin
    x_center = 0.5
    y_center = 0.5
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _convert_split(df, split_name: str, yolo_root: Path, bbox_margin: float,
                    use_symlinks: bool = True) -> int:
    """
    Convertit un split (train/val/test) du DataFrame vers la structure YOLO.

    Args:
        df           : DataFrame avec colonnes 'filepath' et 'label_idx'
        split_name   : "train" | "val" | "test"
        yolo_root    : racine du dataset YOLO (data/processed/yolo_dataset)
        bbox_margin  : marge (%) retirée des bords pour la bbox 'full_image'
        use_symlinks : si True, crée des liens symboliques (rapide, économe
                       en espace disque) au lieu de copier les images.

    Returns:
        Nombre d'images converties avec succès
    """
    img_dir   = yolo_root / "images" / split_name
    label_dir = yolo_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    n_converted = 0
    n_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}",
                       unit="img"):
        src_path = Path(row["filepath"])
        if not src_path.exists():
            n_skipped += 1
            continue

        class_idx = int(row["label_idx"])

        # Nom de fichier unique (évite les collisions entre classes)
        dest_name = f"{class_idx}_{src_path.stem}{src_path.suffix}"
        dest_img_path = img_dir / dest_name
        dest_label_path = label_dir / f"{class_idx}_{src_path.stem}.txt"

        # Image (lien symbolique par défaut, copie si demandé/impossible)
        if not dest_img_path.exists():
            try:
                if use_symlinks:
                    dest_img_path.symlink_to(src_path.resolve())
                else:
                    shutil.copy(src_path, dest_img_path)
            except OSError:
                # Fallback copie si symlink non supporté (ex: certains FS Windows)
                shutil.copy(src_path, dest_img_path)

        # Label YOLO (bbox = image entière avec marge)
        label_line = _make_full_image_label(class_idx, bbox_margin)
        dest_label_path.write_text(label_line + "\n")

        n_converted += 1

    if n_skipped:
        print(f"    ⚠️  {n_skipped} image(s) introuvable(s), ignorée(s)")

    return n_converted


def build_yolo_dataset(split_result: dict, config: dict) -> str:
    """
    Pipeline complet : convertit train_df/val_df/test_df vers le format
    YOLO et génère le fichier data.yaml attendu par Ultralytics.

    Args:
        split_result : résultat de split_dataset() (Phase 1) — doit contenir
                       'train_df', 'val_df', 'test_df', 'idx2label'
        config       : configuration globale

    Returns:
        Chemin (str) vers le fichier data.yaml généré, à passer directement
        à model.train(data=...)
    """
    print("\n" + "=" * 55)
    print("  Préparation du dataset YOLO (mode bbox = full_image)")
    print("=" * 55)

    det_cfg     = config["phase2"]["detection"]
    bbox_mode   = det_cfg.get("bbox_mode", "full_image")
    bbox_margin = det_cfg.get("bbox_margin", 0.05)
    yolo_root   = Path(config["phase2"]["paths"]["yolo_dataset"])

    if bbox_mode != "full_image":
        raise NotImplementedError(
            f"❌ bbox_mode='{bbox_mode}' non supporté. Seul 'full_image' "
            f"est implémenté (pas de bounding boxes réelles disponibles). "
            f"Pour utiliser de vraies annotations, fournis un dataset déjà "
            f"au format YOLO et passe directement son data.yaml à "
            f"l'entraînement, sans passer par ce module."
        )

    train_df = split_result["train_df"]
    val_df   = split_result["val_df"]
    test_df  = split_result["test_df"]
    idx2label = split_result["idx2label"]

    label_names = [idx2label[i] for i in range(len(idx2label))]

    print(f"\n  ⚠️  Mode bbox : 'full_image' (marge {bbox_margin*100:.0f}%)")
    print(f"     Pas de bounding boxes réelles disponibles dans ce dataset.")
    print(f"     YOLO apprendra donc classification + localisation grossière")
    print(f"     (bbox = quasi-totalité de l'image). À remplacer par de")
    print(f"     vraies annotations dès qu'un dataset annoté est disponible.\n")

    print(f"  📁 Destination : {yolo_root}")
    print(f"  🗂️  Classes ({len(label_names)}) : {label_names}\n")

    # Nettoyage d'une éventuelle conversion précédente
    if yolo_root.exists():
        print(f"  ℹ️  Dataset YOLO existant détecté — reconstruction...")
        shutil.rmtree(yolo_root)

    print("\n[1/4] Conversion du split train...")
    n_train = _convert_split(train_df, "train", yolo_root, bbox_margin)
    print(f"  ✅ {n_train} images converties")

    print("\n[2/4] Conversion du split val...")
    n_val = _convert_split(val_df, "val", yolo_root, bbox_margin)
    print(f"  ✅ {n_val} images converties")

    print("\n[3/4] Conversion du split test...")
    n_test = _convert_split(test_df, "test", yolo_root, bbox_margin)
    print(f"  ✅ {n_test} images converties")

    # Génération du data.yaml (format attendu par Ultralytics)
    print("\n[4/4] Génération de data.yaml...")
    data_yaml = {
        "path" : str(yolo_root.resolve()),
        "train": "images/train",
        "val"  : "images/val",
        "test" : "images/test",
        "nc"   : len(label_names),
        "names": label_names,
    }
    data_yaml_path = yolo_root / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml_lib.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✅ data.yaml généré : {data_yaml_path}")
    print(f"\n  📊 Résumé : {n_train} train / {n_val} val / {n_test} test images")
    print("  ✅ Dataset YOLO prêt\n")

    return str(data_yaml_path)