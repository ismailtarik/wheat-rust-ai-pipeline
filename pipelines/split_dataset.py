import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


CLASS_COLORS = {
    "Yellow_Rust": "#F4C542",
    "Brown_Rust":  "#8B4513",
    "Stem_Rust":   "#C0392B",
    "Healthy":     "#27AE60",
}


def _stratified_split(df: pd.DataFrame, train_ratio: float,
                       val_ratio: float, seed: int) -> tuple:
    """
    Effectue le split stratifié en 3 sous-ensembles.

    Args:
        df          : DataFrame complet avec colonne 'label_idx'
        train_ratio : proportion pour l'entraînement (ex: 0.70)
        val_ratio   : proportion pour la validation (ex: 0.15)
        seed        : graine aléatoire

    Returns:
        (train_df, val_df, test_df)
    """
    # Étape 1 : train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label_idx"],
        random_state=seed
    )

    # Étape 2 : val vs test (50/50 du reste → 15% / 15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_idx"],
        random_state=seed
    )

    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True), \
           test_df.reset_index(drop=True)


def _plot_split_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame,
                              test_df: pd.DataFrame, label_names: list,
                              save_path: Path) -> None:
    """
    Génère un graphique groupé de la distribution par split.
    """
    splits = {
        "Train": train_df["label"].value_counts(),
        "Val":   val_df["label"].value_counts(),
        "Test":  test_df["label"].value_counts(),
    }
    summary = pd.DataFrame(splits).fillna(0).astype(int)
    summary = summary.reindex(label_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(label_names))
    width  = 0.25
    colors = ["#2ECC71", "#3498DB", "#E74C3C"]

    for i, (split_name, color) in enumerate(zip(splits.keys(), colors)):
        bars = ax.bar(
            x + i * width, summary[split_name],
            width=width, label=split_name,
            color=color, edgecolor="white", linewidth=1.2
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 2, str(int(h)),
                    ha="center", va="bottom", fontsize=8, fontweight="bold"
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels(label_names, fontsize=10)
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution par classe et par split (Train / Val / Test)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def split_dataset(validation_result: dict, config: dict) -> dict:
    """
    Pipeline complet de split des données.

    Args:
        validation_result : Résultat de validate_dataset() — doit contenir
                            'df', 'label2idx', 'idx2label', 'class_weights'
        config            : Dictionnaire chargé depuis config.yaml

    Returns:
        dict avec :
            - train_df      : DataFrame d'entraînement
            - val_df        : DataFrame de validation
            - test_df       : DataFrame de test
            - label2idx     : dict { label: index }
            - idx2label     : dict { index: label }
            - class_weights : dict { index: poids }
    """
    print("=" * 55)
    print("  ÉTAPE 4 — Split Stratifié (Train / Val / Test)")
    print("=" * 55)

    seed         = config["project"]["seed"]
    train_ratio  = config["split"]["train_ratio"]
    val_ratio    = config["split"]["val_ratio"]
    processed_dir = Path(config["paths"]["processed"])
    reports_dir   = Path(config["paths"]["reports"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df         = validation_result["df"]
    label2idx  = validation_result["label2idx"]
    idx2label  = validation_result["idx2label"]
    label_names = sorted(label2idx.keys())

    print(f"\n  Paramètres du split :")
    print(f"    Train : {train_ratio*100:.0f}%")
    print(f"    Val   : {val_ratio*100:.0f}%")
    print(f"    Test  : {(1-train_ratio-val_ratio)*100:.0f}%")
    print(f"    Seed  : {seed}")
    print(f"    Stratification :")

    # 1. Split
    print("\n[1/4] Split stratifié en cours...")
    train_df, val_df, test_df = _stratified_split(df, train_ratio, val_ratio, seed)

    n_total = len(df)
    print(f"\n  {'Split':<8} {'Images':>8} {'%':>7}")
    print(f"  {'-'*25}")
    print(f"  {'Train':<8} {len(train_df):>8} {len(train_df)/n_total*100:>6.1f}%")
    print(f"  {'Val':<8} {len(val_df):>8}   {len(val_df)/n_total*100:>6.1f}%")
    print(f"  {'Test':<8} {len(test_df):>8}   {len(test_df)/n_total*100:>6.1f}%")
    print(f"  {'-'*25}")
    print(f"  {'Total':<8} {n_total:>8}")

    # Vérification de la stratification
    print(f"\n  Vérification stratification :")
    check = pd.DataFrame({
        "Train": train_df["label"].value_counts(),
        "Val":   val_df["label"].value_counts(),
        "Test":  test_df["label"].value_counts(),
    }).fillna(0).astype(int)
    print(check.to_string(col_space=8))

    # 2. Recalcul des class weights sur train uniquement
    print("\n[2/4] Calcul des poids de classe (sur train)...")
    cw_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_names)),
        y=train_df["label_idx"].values
    )
    class_weights = {i: float(w) for i, w in enumerate(cw_array)}

    print(f"  Poids de classe :")
    for idx, w in class_weights.items():
        print(f"    [{idx}] {idx2label[idx]:<15} → {w:.4f}")

    # 3. Sauvegarde CSV
    print("\n[3/4] Sauvegarde des CSV...")
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir   / "val.csv",   index=False)
    test_df.to_csv(processed_dir  / "test.csv",  index=False)
    print(f"  train.csv  ({len(train_df)} lignes)")
    print(f"  val.csv    ({len(val_df)} lignes)")
    print(f"  test.csv   ({len(test_df)} lignes)")

    # Sauvegarde metadata JSON
    metadata = {
        "thesis"       : "Deep Learning, CV & IoT for Epidemic Forecasting in Agriculture",
        "author"       : config["project"]["author"],
        "lab"          : config["project"]["lab"],
        "phase"        : "Phase 1 — Dataset Acquisition & Preprocessing",
        "dataset"      : "PlantVillage",
        "target"       : "Wheat Rust (Yellow / Brown / Stem)",
        "framework"    : "TensorFlow / Keras",
        "img_size"     : config["preprocessing"]["img_size"],
        "num_classes"  : config["classes"]["num_classes"],
        "label2idx"    : label2idx,
        "idx2label"    : {str(k): v for k, v in idx2label.items()},
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "splits": {
            "train": len(train_df),
            "val"  : len(val_df),
            "test" : len(test_df),
            "total": n_total,
        },
        "augmentation" : config["augmentation"],
        "normalization": "divide by 255 → [0, 1]",
        "seed"         : seed,
    }
    meta_path = Path(config["paths"]["metadata_json"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  metadata.json")

    # 4. Figure de distribution par split
    print("\n[4/4] Génération de la figure de distribution...")
    _plot_split_distribution(
        train_df, val_df, test_df, label_names,
        save_path=reports_dir / "05_split_distribution.png"
    )
    print(f"  05_split_distribution.png")

    print("\n  Split terminé — prêt pour la Phase 2 (Modèles Baseline)\n")

    return {
        "train_df"     : train_df,
        "val_df"       : val_df,
        "test_df"      : test_df,
        "label2idx"    : label2idx,
        "idx2label"    : idx2label,
        "class_weights": class_weights,
    }