"""
pipelines/train_detection.py
-------------------------------
Phase 2 (Detection) -- Entrainement et evaluation YOLO.

Entraine un modele Ultralytics YOLO (YOLO11n par defaut) sur le dataset
converti par prepare_yolo_dataset.py (bbox = full_image en l'absence
d'annotations reelles).

Usage :
    from pipelines.train_detection import run_detection_phase
    results = run_detection_phase(split_result, config)
"""

import json
from pathlib import Path

from pipelines.prepare_yolo_dataset import build_yolo_dataset


# ─────────────────────────────────────────────────────────────
# Entrainement YOLO
# ─────────────────────────────────────────────────────────────

def _train_yolo(data_yaml_path: str, config: dict, output_dir: Path):
    """
    Entraine le modele YOLO sur le dataset converti.
    L'import d'ultralytics est fait ici (lazy) pour eviter
    l'echec du module entier si ultralytics n'est pas installe.
    """
    from ultralytics import YOLO

    det_cfg = config["phase2"]["detection"]

    print(f"\n  Chargement du modele pre-entraine : {det_cfg['model']}")
    model = YOLO(det_cfg["model"])

    print(f"\n  Configuration d'entrainement :")
    print(f"    epochs : {det_cfg['epochs']}")
    print(f"    imgsz  : {det_cfg['imgsz']}")
    print(f"    batch  : {det_cfg['batch']}")

    train_results = model.train(
        data=data_yaml_path,
        epochs=det_cfg["epochs"],
        imgsz=det_cfg["imgsz"],
        batch=det_cfg["batch"],
        project=str(output_dir),
        name="yolo_train",
        exist_ok=True,
        verbose=True,
    )

    return model, train_results


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def _evaluate_yolo(model, data_yaml_path: str, output_dir: Path) -> dict:
    """
    Evalue le modele entraine sur le split test et extrait les metriques
    cles (mAP50, mAP50-95, precision, recall), globales et par classe.
    """
    print("\n  Evaluation sur le split test...")

    val_results = model.val(
        data=data_yaml_path,
        split="test",
        project=str(output_dir),
        name="yolo_eval",
        exist_ok=True,
    )

    metrics = {
        "mAP50"    : float(val_results.box.map50),
        "mAP50_95" : float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall"   : float(val_results.box.mr),
    }

    print(f"\n  mAP50     : {metrics['mAP50']:.4f}")
    print(f"  mAP50-95  : {metrics['mAP50_95']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")

    # Metriques par classe
    per_class = {}
    try:
        class_names  = val_results.names
        ap50_per_class = val_results.box.ap50
        for idx, ap50 in enumerate(ap50_per_class):
            class_name = class_names.get(idx, f"class_{idx}")
            per_class[class_name] = round(float(ap50), 4)
    except Exception as e:
        print(f"  Metriques par classe non disponibles ({e})")

    metrics["per_class_mAP50"] = per_class
    return metrics


def _plot_per_class_map(per_class_map: dict, save_path: Path) -> None:
    """Genere un graphique en barres du mAP50 par classe (import matplotlib lazy)."""
    if not per_class_map:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sorted_items = sorted(per_class_map.items(), key=lambda x: x[1])
        labels = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]

        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.4)))
        colors = ["#E74C3C" if v < 0.5 else "#F39C12" if v < 0.75 else "#2ECC71"
                 for v in values]
        bars = ax.barh(labels, values, color=colors, edgecolor="white")

        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=9)

        ax.set_xlabel("mAP@50")
        ax.set_xlim(0, 1.05)
        ax.set_title("mAP@50 par classe -- YOLO Detection (bbox = full_image)",
                    fontsize=12, fontweight="bold")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure sauvegardee : {save_path}")
    except Exception as e:
        print(f"  Figure non generee ({e})")


# ─────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────

def run_detection_phase(split_result: dict, config: dict,
                         skip_existing: bool = True) -> dict:
    """
    Pipeline complet de la Phase 2 (Detection) :
        1. Conversion du dataset vers le format YOLO (bbox = full_image)
        2. Entrainement YOLO
        3. Evaluation (mAP, precision, recall -- globale et par classe)
        4. Sauvegarde des resultats et figures

    Args:
        split_result  : resultat de split_dataset() (Phase 1)
        config        : configuration globale
        skip_existing : si True et qu'un modele entraine existe deja
                        (best.pt + metrics.json), le recharge sans
                        re-entrainer (utile apres une coupure de session)

    Returns:
        dict avec les metriques et chemins des resultats
    """
    print("\n" + "=" * 55)
    print("  PHASE 2 (Detection) -- Entrainement YOLO")
    print("=" * 55)

    output_dir = Path(config["phase2"]["paths"]["detection_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path      = output_dir / "metrics.json"
    best_weights_path = output_dir / "yolo_train" / "weights" / "best.pt"

    # Reprise automatique si deja entraine
    if skip_existing and metrics_path.exists() and best_weights_path.exists():
        print(f"\n  Modele YOLO deja entraine -- rechargement depuis le disque")
        print(f"  {best_weights_path}")
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"\n  mAP50: {metrics['mAP50']:.4f}  "
              f"mAP50-95: {metrics['mAP50_95']:.4f}")
        return {
            "metrics"     : metrics,
            "weights_path": str(best_weights_path),
            "output_dir"  : str(output_dir),
        }

    # 1. Conversion du dataset
    data_yaml_path = build_yolo_dataset(split_result, config)

    # 2. Entrainement
    print("\n" + "=" * 55)
    print("  Entrainement YOLO en cours...")
    print("=" * 55)
    model, _ = _train_yolo(data_yaml_path, config, output_dir)

    # 3. Evaluation
    metrics = _evaluate_yolo(model, data_yaml_path, output_dir)

    # 4. Sauvegarde
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metriques sauvegardees : {metrics_path}")

    _plot_per_class_map(metrics["per_class_mAP50"],
                        output_dir / "per_class_map50.png")

    if best_weights_path.exists():
        print(f"  Meilleurs poids : {best_weights_path}")

    print("\n" + "=" * 55)
    print("  PHASE 2 (Detection) terminee")
    print("=" * 55)
    print(f"\n  Rappel : bbox = 'full_image' (pas d'annotations reelles).")
    print(f"  Les metriques mAP refletent surtout la classification.\n")

    return {
        "metrics"     : metrics,
        "weights_path": str(best_weights_path),
        "output_dir"  : str(output_dir),
    }