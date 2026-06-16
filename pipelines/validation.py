import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Backend non-interactif (Colab safe)

from pathlib import Path
from collections import Counter


# Palette de couleurs pour les classes connues (Wheat Rust historique).
# Pour toute classe non listée ici (mode auto avec noms différents),
# une couleur est générée automatiquement — voir _get_class_colors().
CLASS_COLORS = {
    "Yellow_Rust": "#F4C542",
    "Brown_Rust":  "#8B4513",
    "Stem_Rust":   "#C0392B",
    "Healthy":     "#27AE60",
}

# Palette de secours pour classes inconnues (cyclique)
_FALLBACK_PALETTE = [
    "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22",
    "#34495E", "#16A085", "#D35400", "#7F8C8D",
    "#2980B9", "#8E44AD", "#27AE60", "#C0392B",
]


def _get_class_colors(label_names: list) -> dict:
    """
    Retourne un mapping {label: couleur} pour TOUTES les classes présentes,
    en réutilisant CLASS_COLORS quand le nom est connu, et en piochant
    dans la palette de secours sinon (mode auto avec noms arbitraires).
    """
    colors = {}
    fallback_idx = 0
    for label in label_names:
        if label in CLASS_COLORS:
            colors[label] = CLASS_COLORS[label]
        else:
            colors[label] = _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]
            fallback_idx += 1
    return colors


def _auto_label_classes(all_classes: dict) -> dict:
    """
    Mode AUTO : utilise directement toutes les classes du dataset comme
    labels, sans filtrage. Le nom de label est dérivé du nom de dossier
    (nettoyage des préfixes type 'Wheat___' ou 'Wheat_').

    Retourne:
        dict { raw_class_name: {"path": str, "count": int, "label": str} }
    """
    wheat_classes = {}
    for cls_name, info in all_classes.items():
        # Nettoyage du nom pour un label plus lisible
        label = cls_name
        for prefix in ("Wheat___", "Wheat__", "Wheat_", "wheat___", "wheat__", "wheat_"):
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        label = label.replace(" ", "_")

        wheat_classes[cls_name] = {
            "path" : info["path"],
            "count": info["count"],
            "label": label
        }
    return wheat_classes


def _filter_wheat_classes(all_classes: dict, label_map: dict) -> dict:
    """
    Filtre les classes Wheat Rust à partir de toutes les classes PlantVillage.

    Retourne:
        dict { raw_class_name: {"path": str, "count": int, "label": str} }
    """
    wheat_classes = {}

    for cls_name, info in all_classes.items():
        normalized = cls_name.lower().replace(" ", "_").replace("-", "_")
        for key, label in label_map.items():
            key_norm = key.lower().replace(" ", "_")
            if key_norm in normalized or normalized == key_norm:
                wheat_classes[cls_name] = {
                    "path" : info["path"],
                    "count": info["count"],
                    "label": label
                }
                break

    return wheat_classes


def _check_image_integrity(image_paths: list, sample_size: int = 200) -> dict:
    """
    Vérifie l'intégrité d'un échantillon d'images.
    Retourne un dict avec les images corrompues et les statistiques de dimensions.
    """
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
    sample = image_paths[:sample_size]

    corrupted = []
    widths, heights = [], []

    for fp in sample:
        try:
            img = cv2.imread(str(fp))
            if img is None:
                corrupted.append(str(fp))
            else:
                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)
        except Exception as e:
            corrupted.append(str(fp))

    return {
        "checked"  : len(sample),
        "corrupted": corrupted,
        "widths"   : widths,
        "heights"  : heights,
    }


def _build_dataframe(wheat_classes: dict) -> pd.DataFrame:
    """
    Construit le DataFrame principal à partir des classes filtrées.
    """
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
    records = []

    for cls_name, info in wheat_classes.items():
        cls_dir = Path(info["path"])
        for img_path in cls_dir.iterdir():
            if img_path.suffix in img_extensions:
                records.append({
                    "filepath" : str(img_path),
                    "class_raw": cls_name,
                    "label"    : info["label"],
                })

    return pd.DataFrame(records)


def _plot_class_distribution(df: pd.DataFrame, save_path: Path,
                              dataset_label: str = "Wheat Disease Dataset") -> None:
    """Génère le graphique de distribution des classes."""
    counts = df["label"].value_counts()
    class_colors = _get_class_colors(counts.index.tolist())
    colors = [class_colors[c] for c in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Distribution des Classes — {dataset_label}",
        fontsize=13, fontweight="bold"
    )

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors,
                       edgecolor="white", linewidth=1.5)
    axes[0].set_title("Nombre d'images par classe")
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Nombre d'images")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + max(counts.values) * 0.01,
            str(val), ha="center", fontweight="bold", fontsize=9
        )

    # Pie chart
    axes[1].pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 8}
    )
    axes[1].set_title("Proportion des classes")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_image_dimensions(widths: list, heights: list, save_path: Path) -> None:
    """Génère le scatter plot des dimensions d'images."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(widths, heights, alpha=0.4, color="#3498DB",
               edgecolors="white", s=40)
    ax.set_xlabel("Largeur (px)")
    ax.set_ylabel("Hauteur (px)")
    ax.set_title("Distribution des dimensions d'images (échantillon)")
    ax.axvline(256, color="red", linestyle="--", label="Cible 256px")
    ax.axhline(256, color="red", linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_sample_images(df: pd.DataFrame, label_names: list, save_path: Path,
                        n_per_class: int = 4, seed: int = 42,
                        dataset_label: str = "Wheat Disease Dataset") -> None:
    """Génère une grille d'exemples d'images par classe."""
    class_colors = _get_class_colors(label_names)

    fig, axes = plt.subplots(
        len(label_names), n_per_class,
        figsize=(n_per_class * 3, len(label_names) * 3)
    )
    fig.suptitle(f"Exemples d'images — {dataset_label}",
                 fontsize=13, fontweight="bold")

    for row_idx, label in enumerate(label_names):
        samples = (
            df[df["label"] == label]["filepath"]
            .sample(min(n_per_class, (df["label"] == label).sum()),
                    random_state=seed)
            .values
        )
        for col_idx in range(n_per_class):
            ax = (axes[row_idx][col_idx]
                  if len(label_names) > 1 else axes[col_idx])
            if col_idx < len(samples):
                img = cv2.imread(samples[col_idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.set_title(
                label if col_idx == 0 else "",
                fontsize=9, fontweight="bold",
                color=class_colors.get(label, "#2C3E50")
            )
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def validate_dataset(collection_result: dict, config: dict) -> dict:
    """
    Pipeline complet de validation et d'analyse exploratoire.

    Args:
        collection_result : Résultat de collect_data()
        config            : Dictionnaire chargé depuis config.yaml

    Returns:
        dict avec :
            - df            : DataFrame complet des images filtrées
            - label2idx     : dict { label: index }
            - idx2label     : dict { index: label }
            - class_weights : dict { index: poids }
            - wheat_classes : dict des classes filtrées
    """
    print("=" * 55)
    print("  ÉTAPE 2 — Validation & Analyse Exploratoire (EDA)")
    print("=" * 55)

    seed         = config["project"]["seed"]
    class_cfg    = config["classes"]
    mode         = class_cfg.get("mode", "manual")
    label_map    = class_cfg.get("label_map", {})
    dataset_label = f"{config['kaggle']['dataset_id'].split('/')[-1]}"
    reports_dir  = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_classes = collection_result["all_classes"]

    # 1. Sélection des classes (mode auto vs manual)
    print(f"\n[1/5] Sélection des classes (mode = '{mode}')...")

    if mode == "auto":
        wheat_classes = _auto_label_classes(all_classes)
    else:
        wheat_classes = _filter_wheat_classes(all_classes, label_map)

    if not wheat_classes:
        print("\n    Aucune classe trouvée.")
        print("  Classes disponibles dans le dataset téléchargé :")
        for cls in all_classes:
            print(f"    {cls}")
        raise ValueError(
            " Aucune classe détectée.\n"
            "   → En mode 'manual', mets à jour 'label_map' dans configs/config.yaml\n"
            "     avec les noms exacts des classes ci-dessus.\n"
            "   → Ou passe en mode 'auto' pour utiliser toutes les classes."
        )

    print(f"  {len(wheat_classes)} classe(s) retenue(s) :\n")
    print(f"  {'Classe raw':<50} {'Label':<20} {'Images':>7}")
    print(f"  {'-'*79}")
    for cls, info in wheat_classes.items():
        print(f"  {cls:<50} {info['label']:<20} {info['count']:>7}")

    # 2. Construction du DataFrame
    print("\n[2/5] Construction du DataFrame...")
    df = _build_dataframe(wheat_classes)

    label_names = sorted(df["label"].unique().tolist())
    label2idx   = {name: i for i, name in enumerate(label_names)}
    idx2label   = {i: name for name, i in label2idx.items()}
    df["label_idx"] = df["label"].map(label2idx)

    print(f"  {len(df)} images indexées — {len(label_names)} classes")

    # Met à jour la config en mémoire avec le nombre réel de classes détecté
    # (utile en mode 'auto' où num_classes n'est pas connu à l'avance).
    config["classes"]["num_classes"] = len(label_names)

    # 3. Vérification d'intégrité
    print("\n[3/5] Vérification de l'intégrité des images...")
    all_paths   = df["filepath"].tolist()
    integrity   = _check_image_integrity(all_paths, sample_size=200)

    if integrity["corrupted"]:
        print(f"    {len(integrity['corrupted'])} image(s) corrompue(s) détectée(s)")
        # Suppression des images corrompues du DataFrame
        df = df[~df["filepath"].isin(integrity["corrupted"])].reset_index(drop=True)
        print(f"  → Supprimées du DataFrame. Reste : {len(df)} images")
    else:
        print(f"  Aucune image corrompue (sur {integrity['checked']} vérifiées)")

    dims = integrity
    if dims["widths"]:
        print(f"  Largeur  — min: {min(dims['widths'])}  "
              f"max: {max(dims['widths'])}  "
              f"moy: {np.mean(dims['widths']):.0f}px")
        print(f"  Hauteur  — min: {min(dims['heights'])}  "
              f"max: {max(dims['heights'])}  "
              f"moy: {np.mean(dims['heights']):.0f}px")

    # 4. Analyse du déséquilibre & class weights
    print("\n[4/5] Analyse du déséquilibre de classes...")
    from sklearn.utils.class_weight import compute_class_weight

    counts = df["label"].value_counts()
    imbalance_ratio = counts.max() / counts.min()

    print(f"  Ratio max/min : {imbalance_ratio:.2f}x → ", end="")
    if imbalance_ratio < 1.5:
        print("Équilibré")
    elif imbalance_ratio < 3.0:
        print("Déséquilibre modéré  — class weighting recommandé")
    else:
        print("Déséquilibre fort — augmentation + class weighting nécessaires")

    cw_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_names)),
        y=df["label_idx"].values
    )
    class_weights = {i: float(w) for i, w in enumerate(cw_array)}

    print(f"\n  Poids de classe :")
    for idx, w in class_weights.items():
        print(f"    [{idx}] {idx2label[idx]:<15} → {w:.4f}")

    # 5. Génération des figures EDA
    print("\n[5/5] Génération des figures EDA...")

    _plot_class_distribution(df, reports_dir / "01_class_distribution.png",
                              dataset_label=dataset_label)
    print(f"  01_class_distribution.png")

    if dims["widths"]:
        _plot_image_dimensions(
            dims["widths"], dims["heights"],
            reports_dir / "02_image_dimensions.png"
        )
        print(f"  02_image_dimensions.png")

    _plot_sample_images(df, label_names, reports_dir / "03_sample_images.png",
                        n_per_class=4, seed=seed, dataset_label=dataset_label)
    print(f"  03_sample_images.png")

    # Sauvegarde du rapport de validation
    report = {
        "total_images"    : len(df),
        "num_classes"     : len(label_names),
        "label2idx"       : label2idx,
        "class_counts"    : counts.to_dict(),
        "imbalance_ratio" : round(imbalance_ratio, 3),
        "class_weights"   : class_weights,
        "corrupted_images": integrity["corrupted"],
        "dim_stats": {
            "width_min" : int(min(dims["widths"])) if dims["widths"] else None,
            "width_max" : int(max(dims["widths"])) if dims["widths"] else None,
            "width_mean": round(float(np.mean(dims["widths"])), 1) if dims["widths"] else None,
            "height_min": int(min(dims["heights"])) if dims["heights"] else None,
            "height_max": int(max(dims["heights"])) if dims["heights"] else None,
            "height_mean":round(float(np.mean(dims["heights"])), 1) if dims["heights"] else None,
        }
    }
    report_path = reports_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Rapport sauvegardé : {report_path}")
    print("\n  Validation terminée — prêt pour preprocessing\n")

    return {
        "df"           : df,
        "label2idx"    : label2idx,
        "idx2label"    : idx2label,
        "class_weights": class_weights,
        "wheat_classes": wheat_classes,
    }