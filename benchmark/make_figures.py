"""EDA et figures de l'article.

Usage : python make_figures.py --list | --stage 12a | --stage 12a,12b | --stage all
"""
import json
import math
import os
import shutil
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
try:
    from IPython.display import display
except ImportError:  # hors notebook
    display = print

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour `import common`




def stage_12a_eda_inventory():
    """EDA : inventaire des images de v3.
    
    Origine : cellule(s) 99 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/classification_v3
    """

    # ══════════ cellule 99 ══════════

    # ============================================================
    # EDA - CLASSIFICATION_V3
    # Étape 1 : Inventaire et caractérisation technique
    # ============================================================


    ImageFile.LOAD_TRUNCATED_IMAGES = False

    ROOT = Path(
        "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    )

    MANIFEST_PATH = ROOT / "classification_manifest.csv"
    REPORT_DIR = ROOT / "eda"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    print("=" * 65)
    print("EDA - CLASSIFICATION_V3")
    print("=" * 65)
    print("Images dans le manifeste :", len(df))
    print("Classes :", df["canonical_class"].nunique())

    # ------------------------------------------------------------
    # 1. Distribution par partition et par classe
    # ------------------------------------------------------------

    distribution = pd.crosstab(
        df["canonical_class"],
        df["new_split"],
        margins=True
    )

    distribution.to_csv(REPORT_DIR / "class_split_distribution.csv")

    print("\nDistribution par classe et partition :")
    display(distribution)

    # ------------------------------------------------------------
    # 2. Inventaire technique des images
    # ------------------------------------------------------------

    records = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyse des images"):

        path = Path(row["v3_destination_path"])

        record = {
            "filename": row["filename"],
            "path": str(path),
            "class": row["canonical_class"],
            "split": row["new_split"],
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "file_size_bytes": None,
            "status": "OK"
        }

        try:
            record["file_size_bytes"] = path.stat().st_size

            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                record["width"], record["height"] = img.size
                record["format"] = img.format
                record["mode"] = img.mode

        except Exception as e:
            record["status"] = "CORRUPTED"
            errors.append({
                "path": str(path),
                "error": str(e)
            })

        records.append(record)

    inventory = pd.DataFrame(records)

    # ------------------------------------------------------------
    # 3. Sauvegarde des résultats
    # ------------------------------------------------------------

    inventory.to_csv(
        REPORT_DIR / "image_inventory.csv",
        index=False
    )

    pd.DataFrame(errors).to_csv(
        REPORT_DIR / "corrupted_images.csv",
        index=False
    )

    # ------------------------------------------------------------
    # 4. Statistiques descriptives
    # ------------------------------------------------------------

    valid = inventory[inventory["status"] == "OK"].copy()

    dimension_stats = valid[
        ["width", "height", "file_size_bytes"]
    ].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])

    dimension_stats.to_csv(
        REPORT_DIR / "image_dimension_statistics.csv"
    )

    format_distribution = valid["format"].value_counts()
    mode_distribution = valid["mode"].value_counts()

    format_distribution.to_csv(
        REPORT_DIR / "format_distribution.csv"
    )

    mode_distribution.to_csv(
        REPORT_DIR / "color_mode_distribution.csv"
    )

    # ------------------------------------------------------------
    # 5. Rapport synthétique
    # ------------------------------------------------------------

    summary = {
        "total_images": int(len(inventory)),
        "valid_images": int(len(valid)),
        "corrupted_images": int(len(errors)),
        "classes": int(df["canonical_class"].nunique()),
        "splits": {
            str(k): int(v)
            for k, v in df["new_split"].value_counts().items()
        },
        "formats": {
            str(k): int(v)
            for k, v in format_distribution.items()
        },
        "color_modes": {
            str(k): int(v)
            for k, v in mode_distribution.items()
        },
        "width": {
            "min": int(valid["width"].min()),
            "max": int(valid["width"].max()),
            "mean": float(valid["width"].mean()),
            "median": float(valid["width"].median())
        },
        "height": {
            "min": int(valid["height"].min()),
            "max": int(valid["height"].max()),
            "mean": float(valid["height"].mean()),
            "median": float(valid["height"].median())
        }
    }

    with open(REPORT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # ------------------------------------------------------------
    # 6. Affichage
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("RÉSULTATS DE L'INVENTAIRE")
    print("=" * 65)

    print("Images valides :", len(valid))
    print("Images corrompues :", len(errors))

    print("\nFormats :")
    display(format_distribution.rename("count").to_frame())

    print("\nModes colorimétriques :")
    display(mode_distribution.rename("count").to_frame())

    print("\nStatistiques dimensions et tailles :")
    display(dimension_stats)

    print("\nRapport sauvegardé dans :", REPORT_DIR)
    print("\nFichiers générés :")
    for p in sorted(REPORT_DIR.iterdir()):
        if p.is_file():
            print("-", p.name)


def stage_12b_eda_enriched_inventory():
    """EDA : inventaire enrichi (eda_enriched_inventory.csv).
    
    Origine : cellule(s) 102 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/classification_v3
    """

    # ══════════ cellule 102 ══════════


    BASE_DIR = "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    EDA_DIR = os.path.join(BASE_DIR, "eda")

    manifest = pd.read_csv(
        os.path.join(BASE_DIR, "classification_manifest.csv")
    )

    inventory = pd.read_csv(
        os.path.join(EDA_DIR, "image_inventory.csv")
    )

    # Extraire le nom du fichier de destination
    manifest["inventory_filename"] = manifest[
        "v3_destination_path"
    ].astype(str).apply(os.path.basename)

    inventory["inventory_filename"] = inventory[
        "path"
    ].astype(str).apply(os.path.basename)

    # Jointure sur le nom SHA-256 du fichier
    data = manifest.merge(
        inventory.drop(columns=["filename"], errors="ignore"),
        on="inventory_filename",
        how="left",
        suffixes=("_manifest", "_inventory"),
        validate="one_to_one"
    )

    # Vérifications
    print("=" * 65)
    print("VALIDATION DE LA JOINTURE")
    print("=" * 65)

    print("Nombre d'images dans le manifeste :", len(manifest))
    print("Nombre de lignes après jointure :", len(data))
    print("Dimensions manquantes :", data["width"].isna().sum())
    print("Hauteurs manquantes :", data["height"].isna().sum())

    print("\nCorrespondance des classes :")
    if "class" in data.columns:
        print(
            "Classes différentes :",
            (
                data["canonical_class"].astype(str)
                != data["class"].astype(str)
            ).sum()
        )

    print("\nCorrespondance des partitions :")
    if "split" in data.columns:
        print(
            "Partitions différentes :",
            (
                data["new_split"].astype(str)
                != data["split"].astype(str)
            ).sum()
        )

    assert len(data) == 12484, "Nombre de lignes inattendu."
    assert data["width"].notna().all(), "Dimensions manquantes."
    assert data["height"].notna().all(), "Hauteurs manquantes."
    assert data["status"].eq("OK").all(), "Images non valides."

    # Vérifier les classes et partitions
    assert (
        data["canonical_class"].astype(str)
        == data["class"].astype(str)
    ).all(), "Incohérence dans les classes."

    assert (
        data["new_split"].astype(str)
        == data["split"].astype(str)
    ).all(), "Incohérence dans les partitions."

    print("\nSUCCESS: jointure validée pour les 12 484 images.")

    # Calculer les statistiques
    data["aspect_ratio"] = data["width"] / data["height"]
    data["resolution"] = data["width"] * data["height"]

    # Sauvegarder le dataset EDA enrichi
    output_path = os.path.join(EDA_DIR, "eda_enriched_inventory.csv")
    data.to_csv(output_path, index=False)

    print("\nFichier sauvegardé :", output_path)

    # Statistiques descriptives
    print("\n=== DIMENSIONS ===")
    print(
        data[["width", "height", "file_size_bytes"]]
        .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
        .round(2)
    )

    print("\n=== ASPECT RATIO ===")
    print(data["aspect_ratio"].describe().round(3))

    print("\n=== RESOLUTION ===")
    print(data["resolution"].describe().round(0))


def stage_12c_eda_figures():
    """EDA : figures finales (fig1 a fig7).
    
    Origine : cellule(s) 103 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/classification_v3
    """

    # ══════════ cellule 103 ══════════

    # ============================================================
    # EDA - STEP 3: FINAL VISUALIZATIONS
    # Dataset: classification_v3
    # ============================================================



    # ------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------

    BASE_DIR = "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"
    EDA_DIR = os.path.join(BASE_DIR, "eda")
    FIG_DIR = os.path.join(EDA_DIR, "figures_final")

    os.makedirs(FIG_DIR, exist_ok=True)

    data = pd.read_csv(
        os.path.join(EDA_DIR, "eda_enriched_inventory.csv")
    )

    # Vérifications
    assert len(data) == 12484
    assert data["width"].notna().all()
    assert data["height"].notna().all()

    data["aspect_ratio"] = data["width"] / data["height"]
    data["resolution"] = data["width"] * data["height"]

    sns.set_theme(style="whitegrid")

    print("Images :", len(data))
    print("Classes :", data["canonical_class"].nunique())

    # ------------------------------------------------------------
    # 2. FIGURE 1: CLASS DISTRIBUTION
    # ------------------------------------------------------------

    class_counts = (
        data["canonical_class"]
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        class_counts.index,
        class_counts.values
    )

    ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_title(
        "Distribution of Images Across the 17 Wheat Classes",
        fontsize=14
    )
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Class")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "fig1_class_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 3. FIGURE 2: SPLIT DISTRIBUTION
    # ------------------------------------------------------------

    split_table = pd.crosstab(
        data["canonical_class"],
        data["new_split"]
    )

    split_order = ["train", "val", "test"]
    split_table = split_table.reindex(
        columns=split_order,
        fill_value=0
    )

    split_pct = split_table.div(
        split_table.sum(axis=1),
        axis=0
    ) * 100

    fig, ax = plt.subplots(figsize=(14, 7))

    split_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(
        "Class Distribution Across Dataset Partitions",
        fontsize=14
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Partition")

    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "fig2_split_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 4. FIGURE 3: IMAGE DIMENSIONS
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        data=data,
        x="width",
        bins=60,
        log_scale=True,
        ax=axes[0]
    )
    axes[0].set_title("Image Width Distribution (Log Scale)")
    axes[0].set_xlabel("Width (pixels, logarithmic scale)")

    sns.histplot(
        data=data,
        x="height",
        bins=60,
        log_scale=True,
        ax=axes[1]
    )
    axes[1].set_title("Image Height Distribution (Log Scale)")
    axes[1].set_xlabel("Height (pixels, logarithmic scale)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "fig3_image_dimensions.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 5. FIGURE 4: ASPECT RATIO
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=data,
        x="aspect_ratio",
        bins=70,
        ax=ax
    )

    median_ratio = data["aspect_ratio"].median()

    ax.axvline(
        median_ratio,
        linestyle="--",
        label=f"Median = {median_ratio:.2f}"
    )

    # Limiter l'affichage aux 99e percentiles pour rendre
    # la distribution principale plus lisible.
    upper_limit = data["aspect_ratio"].quantile(0.99)
    ax.set_xlim(0, upper_limit)

    ax.set_title("Image Aspect Ratio Distribution")
    ax.set_xlabel("Width / Height")
    ax.set_ylabel("Number of Images")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "fig4_aspect_ratio.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 6. FIGURE 5: RESOLUTION DISTRIBUTION
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=data,
        x="resolution",
        bins=60,
        log_scale=True,
        ax=ax
    )

    ax.set_title("Image Resolution Distribution (Log Scale)")
    ax.set_xlabel("Width × Height (pixels, logarithmic scale)")
    ax.set_ylabel("Number of Images")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "fig5_resolution_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 7. FIGURE 6: REPRESENTATIVE CLASS IMAGES
    # ------------------------------------------------------------

    classes = sorted(data["canonical_class"].unique())

    ncols = 4
    nrows = math.ceil(len(classes) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, nrows * 4)
    )

    axes = np.array(axes).reshape(-1)

    for ax, class_name in zip(axes, classes):

        candidates = data[
            data["canonical_class"] == class_name
        ]

        sample = candidates.sample(
            n=1,
            random_state=42
        ).iloc[0]

        image_path = sample["v3_destination_path"]

        try:
            with Image.open(image_path) as img:
                ax.imshow(img.convert("RGB"))

            ax.set_title(class_name, fontsize=10)
            ax.axis("off")

        except Exception as error:
            print("Erreur image:", image_path, error)
            ax.set_title(class_name + " - Read error")
            ax.axis("off")

    for ax in axes[len(classes):]:
        ax.axis("off")

    plt.suptitle(
        "Representative Samples from the 17 Wheat Classes",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "fig6_class_examples.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 8. FIGURE 7: FORMAT AND COLOR MODE
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    data["format"].value_counts().plot(
        kind="bar",
        ax=axes[0]
    )
    axes[0].set_title("Image Format Distribution")
    axes[0].set_xlabel("Format")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=30)

    data["mode"].value_counts().plot(
        kind="bar",
        ax=axes[1]
    )
    axes[1].set_title("Color Mode Distribution")
    axes[1].set_xlabel("Color Mode")
    axes[1].set_ylabel("Number of Images")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "fig7_format_color_mode.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ------------------------------------------------------------
    # 9. SAVE FINAL STATISTICS
    # ------------------------------------------------------------

    summary = {
        "number_of_images": len(data),
        "number_of_classes": data["canonical_class"].nunique(),
        "min_width": int(data["width"].min()),
        "max_width": int(data["width"].max()),
        "median_width": float(data["width"].median()),
        "min_height": int(data["height"].min()),
        "max_height": int(data["height"].max()),
        "median_height": float(data["height"].median()),
        "median_aspect_ratio": float(data["aspect_ratio"].median()),
        "min_resolution": int(data["resolution"].min()),
        "max_resolution": int(data["resolution"].max()),
        "median_resolution": float(data["resolution"].median()),
        "imbalance_ratio": float(
            data["canonical_class"].value_counts().max()
            / data["canonical_class"].value_counts().min()
        )
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    summary_df.to_csv(
        os.path.join(EDA_DIR, "final_eda_summary.csv"),
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL EDA VISUALIZATION COMPLETED")
    print("=" * 65)

    print(summary_df.to_string(index=False))
    print("\nFigures saved in:", FIG_DIR)


def stage_18_paper_figures():
    """Figures de l'article (matrice de confusion, etc.).
    
    Origine : cellule(s) 151 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/drive/MyDrive/Colab Notebooks/ ;  ; data/processed/classification_v3/test ; /content/FIGURES_PAPIER ; /content/FIGURES_ET_TABLEAUX_CORRIGES
    """

    # ══════════ cellule 151 ══════════

    # ============================================================
    # GENERATION DES FIGURES 2, 3 ET 4 POUR LE PAPIER SCIENTIFIQUE
    # ResNet50 - Wheat Disease Classification
    # ============================================================


    # ------------------------------------------------------------
    # 1. CONFIGURATION : ADAPTER CES CHEMINS
    # ------------------------------------------------------------

    BASE = Path(
        "/content/drive/MyDrive/Colab Notebooks/"
    "WHEAT_RUST_RESEARCH1/03_EXPERIMENTS/"
    "RECOVERED_RESNET50_BASELINE"
    )

    # Rechercher automatiquement test_predictions.csv
    candidates = list(BASE.rglob("test_predictions.csv"))

    if not candidates:
        raise FileNotFoundError(
            "test_predictions.csv introuvable. "
        "Vérifiez le chemin BASE."
        )

    PREDICTIONS_CSV = candidates[0]
    print("Prédictions utilisées :", PREDICTIONS_CSV)

    # Dossier contenant les images du jeu de test.
    # Adaptez-le au chemin réel de votre benchmark.
    TEST_DIR = Path(
        "/content/wheat-rust-ai-pipeline/"
    "data/processed/classification_v3/test"
    )

    OUT_DIR = Path("/content/FIGURES_PAPIER")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 2. CHARGEMENT ET VERIFICATION DES PREDICTIONS
    # ------------------------------------------------------------

    df = pd.read_csv(PREDICTIONS_CSV)
    print("\nColonnes disponibles :", df.columns.tolist())
    print("Nombre de prédictions :", len(df))

    required = {"true_label", "predicted_label"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Le CSV doit contenir true_label et predicted_label. "
        f"Colonnes disponibles : {df.columns.tolist()}"
        )

    df["true_label"] = df["true_label"].astype(str)
    df["predicted_label"] = df["predicted_label"].astype(str)

    # Si le CSV possède une colonne correct, la contrôler.
    if "correct" in df.columns:
        reported_correct = (
            df["correct"].astype(str).str.lower()
            .isin(["true", "1", "yes"])
        )
        actual_correct = df["true_label"].eq(df["predicted_label"])

        if not np.array_equal(
            reported_correct.to_numpy(),
            actual_correct.to_numpy()
        ):
            warnings.warn(
                "La colonne correct ne correspond pas entièrement "
            "aux labels vrai/prédit."
            )

    classes = sorted(df["true_label"].unique())

    if len(classes) != 17:
        warnings.warn(
            f"{len(classes)} classes vraies détectées au lieu de 17. "
        "Vérifiez le CSV avant d'utiliser les figures."
        )

    # Ordre commun pour les figures et les métriques
    class_names = classes

    report = classification_report(
        df["true_label"],
        df["predicted_label"],
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    print("\nAccuracy recalculée :",
          round((df["true_label"] == df["predicted_label"]).mean(), 6))

    # ------------------------------------------------------------
    # FIGURE 2 : F1-SCORE PAR CLASSE
    # ------------------------------------------------------------

    f1_scores = pd.Series({
        cls: report[cls]["f1-score"]
        for cls in class_names
    }).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(
        f1_scores.index,
        f1_scores.values,
        edgecolor="black",
        linewidth=0.35
    )

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Canonical wheat disease class")
    ax.set_title("ResNet50 Class-wise F1-score")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, score in zip(bars, f1_scores.values):
        ax.text(
            min(score + 0.012, 1.015),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8
        )

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "Figure_2_Classwise_F1.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Export du tableau complet associé à la figure 2
    report_df = pd.DataFrame({
        "Class": class_names,
        "Precision": [report[c]["precision"] for c in class_names],
        "Recall": [report[c]["recall"] for c in class_names],
        "F1_score": [report[c]["f1-score"] for c in class_names],
        "Support": [int(report[c]["support"]) for c in class_names]
    })

    report_df.to_csv(
        OUT_DIR / "Corrected_Table_5.csv",
        index=False
    )

    # ------------------------------------------------------------
    # FIGURE 3 : MATRICE DE CONFUSION NORMALISEE PAR LIGNE
    # ------------------------------------------------------------

    cm = confusion_matrix(
        df["true_label"],
        df["predicted_label"],
        labels=class_names
    )

    row_totals = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(
        cm_normalized,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=1
    )

    fig.colorbar(im, ax=ax, label="Proportion within true class")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Ground-truth class",
        title="Row-normalized Confusion Matrix - ResNet50"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=8
    )
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Afficher les proportions dans les cellules
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_normalized[i, j]
            if value >= 0.04 or cm[i, j] > 0:
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if value < 0.25 else "black"
                )

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "Figure_3_Confusion_Matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Export des matrices numérique et normalisée
    pd.DataFrame(
        cm, index=class_names, columns=class_names
    ).to_csv(OUT_DIR / "Confusion_Matrix_Counts.csv")

    pd.DataFrame(
        cm_normalized, index=class_names, columns=class_names
    ).to_csv(OUT_DIR / "Confusion_Matrix_Normalized.csv")

    # ------------------------------------------------------------
    # TABLEAU 6 : CONFUSIONS LES PLUS FREQUENTES
    # ------------------------------------------------------------

    confusions = []
    for i, true_class in enumerate(class_names):
        for j, predicted_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "Ground-truth class": true_class,
                    "Predicted class": predicted_class,
                    "Errors": int(cm[i, j])
                })

    confusions_df = pd.DataFrame(confusions)
    confusions_df = confusions_df.sort_values(
        "Errors", ascending=False
    )

    confusions_df.to_csv(
        OUT_DIR / "Corrected_Table_6_All_Confusions.csv",
        index=False
    )

    print("\nTop 15 confusions :")
    print(confusions_df.head(15).to_string(index=False))

    # ------------------------------------------------------------
    # FIGURE 4 : EXEMPLES D'ERREURS QUALITATIVES
    # ------------------------------------------------------------

    # Les cinq motifs principaux observés dans l'archive.
    patterns = [
        ("Tan_Spot", "Leaf_Blight"),
        ("Mite", "Aphid"),
        ("Leaf_Blight", "Tan_Spot"),
        ("Common_Root_Rot", "Mite"),
        ("Black_Rust", "Brown_Rust")
    ]

    # Chercher une colonne contenant le chemin de l'image
    path_candidates = [
        "image_path", "filepath", "file_path",
        "path", "image", "filename", "file_name"
    ]
    path_col = next(
        (c for c in path_candidates if c in df.columns),
        None
    )

    # Construire un index nom de fichier -> chemin
    image_index = {}
    if TEST_DIR.exists():
        for p in TEST_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp"
            }:
                image_index[p.name] = p

    def find_image(row):
        # Première possibilité : chemin complet stocké dans le CSV
        if path_col is not None:
            raw = str(row[path_col])
            p = Path(raw)
            if p.is_file():
                return p

            # Sinon, recherche par nom de fichier
            if p.name in image_index:
                return image_index[p.name]

        # Deuxième possibilité : recherche à partir d'un identifiant
        for col in ["image_id", "filename", "file_name", "image"]:
            if col in row.index:
                name = Path(str(row[col])).name
                if name in image_index:
                    return image_index[name]

        return None

    # Extraire jusqu'à trois erreurs pour chacun des cinq motifs
    selected = []
    for true_cls, pred_cls in patterns:
        subset = df[
            (df["true_label"] == true_cls) &
            (df["predicted_label"] == pred_cls)
        ]

        count_added = 0
        for _, row in subset.iterrows():
            img_path = find_image(row)
            if img_path is not None:
                selected.append({
                    "path": img_path,
                    "true": true_cls,
                    "pred": pred_cls,
                    "confidence": row.get("confidence", np.nan)
                })
                count_added += 1

            if count_added == 3:
                break

        print(
            f"{true_cls} -> {pred_cls}: "
        f"{len(subset)} erreurs, {count_added} images trouvées"
        )

    if not selected:
        print(
            "\nAucune image retrouvée pour la figure 4. "
        "Vérifiez TEST_DIR et les colonnes du CSV."
        )
    else:
        ncols = 3
        nrows = int(np.ceil(len(selected) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(12, 4 * nrows)
        )
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for ax, item in zip(axes, selected):
            try:
                with Image.open(item["path"]) as im:
                    ax.imshow(im.convert("RGB"))

                title = (
                    f"True: {item['true']}\n"
                f"Predicted: {item['pred']}"
                )

                if pd.notna(item["confidence"]):
                    title += f"\nConfidence: {float(item['confidence']):.3f}"

                ax.set_title(title, fontsize=9)
                ax.axis("off")

            except Exception as e:
                ax.set_title(f"Image unavailable: {e}", fontsize=8)
                ax.axis("off")

        fig.suptitle(
            "Representative ResNet50 Classification Errors",
            fontsize=14
        )
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / "Figure_4_Qualitative_Errors.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    # ------------------------------------------------------------
    # 4. EXPORT FINAL
    # ------------------------------------------------------------

    print("\nFichiers générés dans :", OUT_DIR)
    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            print(p.name, "-", round(p.stat().st_size / 1024, 1), "KB")

    # Télécharger toutes les figures et les tableaux
    from google.colab import files

    ZIP_FIGURES = shutil.make_archive(
        "/content/FIGURES_ET_TABLEAUX_CORRIGES",
        "zip",
        OUT_DIR
    )

    files.download(ZIP_FIGURES)


STAGES = {
    "12a": stage_12a_eda_inventory,
    "12b": stage_12b_eda_enriched_inventory,
    "12c": stage_12c_eda_figures,
    "18": stage_18_paper_figures,
}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--stage", default="all", help="numero(s) d'etape separes par des virgules, ou 'all'")
    ap.add_argument("--list", action="store_true", help="lister les etapes")
    a = ap.parse_args(argv)
    if a.list:
        for k, f in STAGES.items(): print(f"{k:>5}  {f.__name__}  -  {(f.__doc__ or '').strip().splitlines()[0]}")
        return
    wanted = list(STAGES) if a.stage == "all" else [x.strip() for x in a.stage.split(",")]
    for k in wanted:
        if k not in STAGES: raise SystemExit(f"Etape inconnue : {k} (voir --list)")
    for k in wanted:
        print(f"\n{'=' * 70}\n  ETAPE {k} : {STAGES[k].__name__}\n{'=' * 70}")
        STAGES[k]()


if __name__ == "__main__":
    main()
