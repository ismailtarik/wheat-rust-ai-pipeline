import os
import shutil
import zipfile
import json
from pathlib import Path


def _setup_kaggle_credentials(kaggle_json_path: str = None) -> None:
    """
    Configure les credentials Kaggle.
    Si kaggle_json_path est fourni, copie le fichier au bon endroit.
    Sinon, suppose qu'il est déjà dans ~/.kaggle/kaggle.json.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    dest = kaggle_dir / "kaggle.json"

    if kaggle_json_path:
        shutil.copy(kaggle_json_path, dest)
        print(f"  kaggle.json copié depuis : {kaggle_json_path}")
    elif dest.exists():
        print(f"  kaggle.json déjà présent : {dest}")
    else:
        raise FileNotFoundError(
            " kaggle.json introuvable.\n"
            "   → Upload ton fichier depuis https://www.kaggle.com/settings → API\n"
            "   → Passe le chemin via kaggle_json_path='chemin/vers/kaggle.json'"
        )

    dest.chmod(0o600)


def _download_dataset(dataset_id: str, download_path: Path) -> None:
    """
    Télécharge et extrait le dataset Kaggle.
    """
    download_path.mkdir(parents=True, exist_ok=True)
    print(f"   Téléchargement de '{dataset_id}'...")

    ret = os.system(
        f"kaggle datasets download -d {dataset_id} "
        f"-p {download_path} --unzip"
    )

    if ret != 0:
        raise RuntimeError(
            f" Échec du téléchargement Kaggle (code {ret}).\n"
            "   Vérifie ton dataset_id et tes credentials."
        )
    print(f"  Dataset extrait dans : {download_path}")


def _detect_image_root(download_path: Path) -> Path:
    """
    Détecte automatiquement le dossier racine contenant les sous-dossiers de classes.
    Cherche le dossier parent des premières images trouvées.
    """
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}

    candidates = {}
    for img_file in download_path.rglob('*'):
        if img_file.suffix in img_extensions and img_file.is_file():
            parent = img_file.parent
            candidates[parent] = candidates.get(parent, 0) + 1

    if not candidates:
        raise FileNotFoundError(
            f" Aucune image trouvée dans {download_path}\n"
            "   Vérifie que l'extraction s'est bien déroulée."
        )

    # Le dossier racine est le parent commun du dossier contenant le plus d'images
    top_dir = max(candidates, key=candidates.get)
    return top_dir.parent


def _scan_dataset(image_root: Path) -> dict:
    """
    Scanne toutes les classes et compte les images disponibles.
    Retourne un dict { nom_classe: {"path": Path, "count": int} }
    """
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
    class_info = {}

    for cls_dir in sorted(image_root.iterdir()):
        if cls_dir.is_dir():
            images = [f for f in cls_dir.iterdir() if f.suffix in img_extensions]
            if images:
                class_info[cls_dir.name] = {
                    "path"  : str(cls_dir),
                    "count" : len(images)
                }

    return class_info


def _has_existing_data(download_path: Path) -> bool:
    """
    Vérifie si download_path contient déjà des images (dataset déjà présent
    localement), pour éviter un téléchargement Kaggle inutile.
    """
    if not download_path.exists():
        return False
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
    for f in download_path.rglob('*'):
        if f.is_file() and f.suffix in img_extensions:
            return True
    return False


def collect_data(config: dict, kaggle_json_path: str = None,
                  skip_download: bool = False) -> dict:
    """
    Pipeline complet de collecte des données.

    Args:
        config          : Dictionnaire chargé depuis config.yaml
        kaggle_json_path: Chemin vers le FICHIER kaggle.json (PAS un dossier !).
                          Ignoré si skip_download=True ou si des données
                          locales sont déjà détectées.
        skip_download   : Si True, force l'utilisation des données locales
                          déjà présentes dans 'paths.raw_data' sans passer
                          par Kaggle.

    Returns:
        dict avec les clés :
            - image_root    : Path vers le dossier des classes
            - all_classes   : dict { nom_classe: {path, count} }
            - total_images  : int
    """
    print("=" * 55)
    print("  ÉTAPE 1 — Collecte des Données (PlantVillage)")
    print("=" * 55)

    dataset_id    = config["kaggle"]["dataset_id"]
    download_path = Path(config["paths"]["raw_data"])

    # Auto-détection : si des images existent déjà localement, on ignore Kaggle
    if not skip_download and _has_existing_data(download_path):
        print(f"\n   Images déjà présentes dans '{download_path}'")
        print(f"      → Téléchargement Kaggle ignoré automatiquement.")
        skip_download = True

    if skip_download:
        print("\n[1/2] Téléchargement ignoré — utilisation des données locales")
    else:
        # 1. Credentials Kaggle
        print("\n[1/3] Configuration Kaggle API...")
        _setup_kaggle_credentials(kaggle_json_path)

        # 2. Téléchargement
        print("\n[2/3] Téléchargement du dataset...")
        _download_dataset(dataset_id, download_path)

    # 3. Scan du contenu (local ou téléchargé)
    step_label = "[2/2]" if skip_download else "[3/3]"
    print(f"\n{step_label} Analyse du contenu...")

    if not download_path.exists():
        raise FileNotFoundError(
            f" Dossier introuvable : {download_path}\n"
            f"   Vérifie 'paths.raw_data' dans configs/config.yaml"
        )

    image_root  = _detect_image_root(download_path)
    all_classes = _scan_dataset(image_root)
    total       = sum(info["count"] for info in all_classes.values())

    print(f"\n  Racine images   : {image_root}")
    print(f"   Classes totales : {len(all_classes)}")
    print(f"   Images totales  : {total}\n")

    # Affichage du résumé
    print(f"  {'Classe':<50} {'Images':>7}")
    print(f"  {'-'*58}")
    for cls_name, info in all_classes.items():
        print(f"  {cls_name:<50} {info['count']:>7}")

    # Sauvegarde du rapport de collecte
    report_path = Path(config["paths"]["reports"]) / "data_collection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_id"   : dataset_id,
        "image_root"   : str(image_root),
        "total_classes": len(all_classes),
        "total_images" : total,
        "classes"      : all_classes
    }
    # Convertir Path en str pour JSON
    report["classes"] = {
        k: {"path": str(v["path"]), "count": v["count"]}
        for k, v in all_classes.items()
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Rapport sauvegardé : {report_path}")
    print("\n  Collecte terminée — prêt pour validation\n")

    return {
        "image_root"  : image_root,
        "all_classes" : all_classes,
        "total_images": total
    }