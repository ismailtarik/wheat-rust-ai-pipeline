"""
pipelines/train_classification.py
------------------------------------
Phase 2 — Étape 2 : Entraînement des modèles de classification.

Entraîne et compare 3 architectures (pipelines/models.py) :
  - cnn_custom      : baseline from scratch
  - resnet50         : transfer learning (2 étapes : head puis fine-tuning)
  - efficientnetb0   : transfer learning (2 étapes : head puis fine-tuning)

Pour chaque modèle :
  1. Construction (build_model)
  2. Compilation (Adam + categorical_crossentropy)
  3. Entraînement de la tête (backbone gelé) avec class weights
  4. Fine-tuning (dégel partiel du backbone, LR réduit)
  5. Évaluation sur le test set (Accuracy, Precision, Recall, F1, matrice de confusion)
  6. Sauvegarde du modèle (.keras) + historique + figures

Usage :
    from pipelines.train_classification import run_classification_phase
    results = run_classification_phase(split_result, train_ds, val_ds, test_ds, config)
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

from pipelines.models import build_model, unfreeze_for_finetuning


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

def _build_callbacks(checkpoint_path: Path, patience_es: int,
                      patience_rlr: int) -> list:
    """Construit les callbacks Keras standards pour l'entraînement."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_rlr,
            min_lr=1e-7,
            verbose=1
        ),
    ]


# ─────────────────────────────────────────────────────────────
# Entraînement d'un modèle (head + fine-tuning)
# ─────────────────────────────────────────────────────────────

def _train_single_model(model_name: str, train_ds, val_ds, test_ds,
                         num_classes: int, class_weights: dict,
                         idx2label: dict, config: dict,
                         output_dir: Path) -> dict:
    """
    Entraîne un modèle complet (head + fine-tuning si applicable)
    et retourne ses métriques + historique.
    """
    cfg         = config["phase2"]["classification"]
    img_size    = tuple(config["preprocessing"]["img_size"])
    input_shape = img_size + (3,)

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  🧠 Modèle : {model_name.upper()}")
    print(f"{'='*55}")

    # ── 1. Construction ──
    is_transfer = model_name != "cnn_custom"
    model = build_model(model_name, input_shape, num_classes,
                         freeze_base=True)
    print(f"  ✅ Architecture construite — {model.count_params():,} paramètres")

    class_weights_arg = class_weights if cfg.get("use_class_weights", True) else None

    history_stage1 = None
    history_stage2 = None
    t_start = time.time()

    # ── 2. Stage 1 : entraînement de la tête (backbone gelé) ──
    print(f"\n  [Stage 1] Entraînement de la tête (backbone gelé)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = _build_callbacks(
        model_dir / f"{model_name}_stage1_best.keras",
        cfg["early_stopping_patience"],
        cfg["reduce_lr_patience"]
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        class_weight=class_weights_arg,
        callbacks=callbacks,
        verbose=1
    )

    # ── 3. Stage 2 : fine-tuning (transfer learning uniquement) ──
    if is_transfer and cfg.get("fine_tune_epochs", 0) > 0:
        print(f"\n  [Stage 2] Fine-tuning (dégel partiel du backbone)...")
        unfreeze_for_finetuning(model, num_layers_to_unfreeze=30)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["fine_tune_lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks_ft = _build_callbacks(
            model_dir / f"{model_name}_stage2_best.keras",
            cfg["early_stopping_patience"],
            cfg["reduce_lr_patience"]
        )

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["fine_tune_epochs"],
            class_weight=class_weights_arg,
            callbacks=callbacks_ft,
            verbose=1
        )

    train_time = time.time() - t_start
    print(f"\n  ⏱️  Temps d'entraînement total : {train_time/60:.1f} min")

    # ── 4. Évaluation sur le test set ──
    print(f"\n  📊 Évaluation sur le test set...")
    metrics = _evaluate_model(model, test_ds, num_classes, idx2label,
                               model_dir, model_name)
    metrics["train_time_min"] = round(train_time / 60, 2)
    metrics["num_params"] = int(model.count_params())

    # ── 5. Sauvegarde modèle final + historique ──
    final_path = model_dir / f"{model_name}_final.keras"
    model.save(final_path)
    print(f"  💾 Modèle sauvegardé : {final_path}")

    _plot_training_history(history_stage1, history_stage2, model_name, model_dir)

    history_dict = {
        "stage1": {k: [float(v) for v in vals] for k, vals in history_stage1.history.items()}
    }
    if history_stage2:
        history_dict["stage2"] = {
            k: [float(v) for v in vals] for k, vals in history_stage2.history.items()
        }
    with open(model_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────

def _evaluate_model(model, test_ds, num_classes: int, idx2label: dict,
                     model_dir: Path, model_name: str) -> dict:
    """
    Évalue le modèle sur le test set : accuracy, precision, recall, f1,
    matrice de confusion, classification report.
    """
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    label_names = [idx2label[i] for i in range(num_classes)]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\n  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")

    report = classification_report(
        y_true, y_pred, target_names=label_names,
        zero_division=0, output_dict=True
    )
    with open(model_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.6), max(6, num_classes * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                cbar_kws={"label": "Nombre de prédictions"})
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(f"Matrice de confusion — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "model_name": model_name,
        "accuracy"  : round(float(accuracy), 4),
        "precision" : round(float(precision), 4),
        "recall"    : round(float(recall), 4),
        "f1_score"  : round(float(f1), 4),
    }


def _plot_training_history(history_stage1, history_stage2, model_name: str,
                            model_dir: Path) -> None:
    """Génère les courbes d'apprentissage (loss & accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Courbes d'apprentissage — {model_name}",
                 fontsize=13, fontweight="bold")

    h1 = history_stage1.history
    epochs1 = range(1, len(h1["loss"]) + 1)

    axes[0].plot(epochs1, h1["loss"], label="Train (stage1)", color="#3498DB")
    axes[0].plot(epochs1, h1["val_loss"], label="Val (stage1)", color="#E74C3C")
    axes[1].plot(epochs1, h1["accuracy"], label="Train (stage1)", color="#3498DB")
    axes[1].plot(epochs1, h1["val_accuracy"], label="Val (stage1)", color="#E74C3C")

    if history_stage2:
        h2 = history_stage2.history
        offset = len(h1["loss"])
        epochs2 = range(offset + 1, offset + len(h2["loss"]) + 1)
        axes[0].plot(epochs2, h2["loss"], label="Train (fine-tune)",
                     color="#2ECC71", linestyle="--")
        axes[0].plot(epochs2, h2["val_loss"], label="Val (fine-tune)",
                     color="#F39C12", linestyle="--")
        axes[1].plot(epochs2, h2["accuracy"], label="Train (fine-tune)",
                     color="#2ECC71", linestyle="--")
        axes[1].plot(epochs2, h2["val_accuracy"], label="Val (fine-tune)",
                     color="#F39C12", linestyle="--")
        axes[0].axvline(offset, color="gray", linestyle=":", alpha=0.6)
        axes[1].axvline(offset, color="gray", linestyle=":", alpha=0.6)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Comparaison finale
# ─────────────────────────────────────────────────────────────

def _plot_model_comparison(all_metrics: list, output_dir: Path) -> None:
    """Génère un graphique comparatif des 3 modèles sur les 4 métriques."""
    df = pd.DataFrame(all_metrics)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.2
    colors = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]

    for i, metric in enumerate(metrics_to_plot):
        bars = ax.bar(x + i * width, df[metric], width=width,
                      label=metric.capitalize(), color=colors[i],
                      edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                   f"{h:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["model_name"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des modèles — Phase 2 Classification",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────

def run_classification_phase(split_result: dict, train_ds, val_ds, test_ds,
                              config: dict) -> dict:
    """
    Pipeline complet : entraîne et compare tous les modèles configurés
    dans phase2.classification.models.

    Args:
        split_result : résultat de split_dataset() (Phase 1)
        train_ds, val_ds, test_ds : pipelines tf.data (Phase 1)
        config       : configuration globale

    Returns:
        dict { model_name: metrics_dict }, plus un résumé comparatif
    """
    print("\n" + "=" * 55)
    print("  PHASE 2 — Entraînement des Modèles Baseline")
    print("=" * 55)

    cfg          = config["phase2"]["classification"]
    num_classes  = config["classes"]["num_classes"]
    idx2label    = split_result["idx2label"]
    class_weights = split_result["class_weights"]
    output_dir   = Path(config["phase2"]["paths"]["classification_outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Modèles à entraîner : {cfg['models']}")
    print(f"  Epochs (stage 1)    : {cfg['epochs']}")
    print(f"  Epochs (fine-tune)  : {cfg.get('fine_tune_epochs', 0)}")
    print(f"  Class weights       : {'activés' if cfg.get('use_class_weights') else 'désactivés'}")

    all_metrics = []
    for model_name in cfg["models"]:
        metrics = _train_single_model(
            model_name, train_ds, val_ds, test_ds,
            num_classes, class_weights, idx2label, config, output_dir
        )
        all_metrics.append(metrics)

    # Comparaison finale
    print("\n" + "=" * 55)
    print("  ✅ RÉSUMÉ COMPARATIF — PHASE 2 CLASSIFICATION")
    print("=" * 55)

    comparison_df = pd.DataFrame(all_metrics)
    print("\n" + comparison_df.to_string(index=False))

    best_model = comparison_df.loc[comparison_df["f1_score"].idxmax(), "model_name"]
    print(f"\n  🏆 Meilleur modèle (F1-score) : {best_model}")

    comparison_df.to_csv(output_dir / "models_comparison.csv", index=False)
    _plot_model_comparison(all_metrics, output_dir)

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump({
            "best_model": best_model,
            "all_results": all_metrics
        }, f, indent=2)

    print(f"\n  📁 Résultats sauvegardés dans : {output_dir}")
    print("  ✅ Phase 2 (Classification) terminée\n")

    return {
        "all_metrics": all_metrics,
        "best_model": best_model,
        "output_dir": str(output_dir),
    }