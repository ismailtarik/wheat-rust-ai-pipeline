"""Entrainement ResNet50 baseline (PyTorch), evaluation, analyse d'erreurs, evaluation par source.

Usage : python train_eval.py --list | --stage 13 | --stage 13,14 | --stage all
"""
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import resnet50, ResNet50_Weights
try:
    from IPython.display import display
except ImportError:  # hors notebook
    display = print

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour `import common`

from common import (PROJECT)

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stage_13_dataset_and_training():
    """Pretraitement (resize avec padding), Dataset PyTorch et entrainement du ResNet50 baseline.
    
    Origine : cellule(s) 104, 106 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/
    """
    global NUM_CLASSES

    # ══════════ cellule 104 ══════════

    # ============================================================
    # STEP 4.1 - CLASSIFICATION PREPROCESSING
    # Dataset: classification_v3
    # Architecture target: ResNet50 pretrained
    # ============================================================



    # ------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------

    SEED = 42
    NUM_WORKERS = 2

    BASE_DIR = (
        "/content/wheat-rust-ai-pipeline/data/processed/"
    "classification_v3"
    )

    EDA_DIR = os.path.join(BASE_DIR, "eda")

    MANIFEST_PATH = os.path.join(
        BASE_DIR, "classification_manifest.csv"
    )

    # Reproductibilité
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    # ------------------------------------------------------------
    # 2. LOAD MANIFEST
    # ------------------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "canonical_class",
        "new_split",
        "v3_destination_path"
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) == 12484

    classes = sorted(df["canonical_class"].unique())

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    idx_to_class = {
        index: class_name
        for class_name, index in class_to_idx.items()
    }

    df["label_idx"] = df["canonical_class"].map(class_to_idx)

    print("\nNumber of images:", len(df))
    print("Number of classes:", len(classes))
    print("Class mapping:")

    for index, class_name in idx_to_class.items():
        print(index, "->", class_name)

    # ------------------------------------------------------------
    # 3. IMAGE PREPROCESSING
    # ------------------------------------------------------------

    def resize_with_padding(image, size=224):
        """
    Resize while preserving aspect ratio.
    Remaining space is filled with neutral gray.
    """

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        # Fit image inside the target square
        image.thumbnail(
            (size, size),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (size, size),
            color=(128, 128, 128)
        )

        x_offset = (size - image.width) // 2
        y_offset = (size - image.height) // 2

        canvas.paste(
            image,
            (x_offset, y_offset)
        )

        return canvas


    # ImageNet normalization for pretrained ResNet50
    image_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ------------------------------------------------------------
    # 4. CUSTOM DATASET
    # ------------------------------------------------------------

    class WheatClassificationDataset(Dataset):

        def __init__(
            self,
            dataframe,
            image_size=224,
            transform=None
        ):

            self.data = dataframe.reset_index(drop=True)
            self.image_size = image_size
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):

            row = self.data.iloc[index]

            image_path = row["v3_destination_path"]
            label = int(row["label_idx"])

            with Image.open(image_path) as img:

                image = resize_with_padding(
                    img,
                    size=self.image_size
                )

            if self.transform is not None:
                image = self.transform(image)

            return image, label


    # ------------------------------------------------------------
    # 5. CREATE TRAIN / VALIDATION / TEST DATASETS
    # ------------------------------------------------------------

    train_df = df[df["new_split"] == "train"].copy()
    val_df = df[df["new_split"] == "val"].copy()
    test_df = df[df["new_split"] == "test"].copy()

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    assert len(train_df) == 8761
    assert len(val_df) == 1859
    assert len(test_df) == 1864

    train_dataset = WheatClassificationDataset(
        train_df,
        image_size=IMAGE_SIZE,
        transform=image_transform
    )

    val_dataset = WheatClassificationDataset(
        val_df,
        image_size=IMAGE_SIZE,
        transform=image_transform
    )

    test_dataset = WheatClassificationDataset(
        test_df,
        image_size=IMAGE_SIZE,
        transform=image_transform
    )

    # ------------------------------------------------------------
    # 6. CREATE DATALOADERS
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ------------------------------------------------------------
    # 7. VERIFY ONE BATCH
    # ------------------------------------------------------------

    images, labels = next(iter(train_loader))

    print("\n" + "=" * 60)
    print("DATALOADER VALIDATION")
    print("=" * 60)

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Image device:", images.device)

    print("Pixel tensor min:", images.min().item())
    print("Pixel tensor max:", images.max().item())

    assert images.shape == (BATCH_SIZE, 3, 224, 224)
    assert labels.shape == (BATCH_SIZE,)

    assert torch.isfinite(images).all()
    assert labels.min() >= 0
    assert labels.max() < len(classes)

    print("\nSUCCESS: preprocessing pipeline validated.")

    # ------------------------------------------------------------
    # 8. VERIFY ALL THREE PARTITIONS
    # ------------------------------------------------------------

    for name, loader in [
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader)
    ]:

        batch_images, batch_labels = next(iter(loader))

        print(
            f"{name}:",
            batch_images.shape,
            "| Labels:",
            batch_labels.shape
        )

    # ------------------------------------------------------------
    # 9. DISPLAY PREPROCESSED IMAGES
    # ------------------------------------------------------------

    # Reverse ImageNet normalization for visualization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        image = images[i].cpu() * std + mean
        image = image.clamp(0, 1)

        image = image.permute(1, 2, 0).numpy()

        label_idx = labels[i].item()
        class_name = idx_to_class[label_idx]

        ax.imshow(image)
        ax.set_title(class_name)
        ax.axis("off")

    plt.suptitle(
        "Preprocessed Wheat Disease Images - 224x224 RGB"
    )

    plt.tight_layout()
    plt.show()

    print("\nSTEP 4.1 COMPLETED")

    # ══════════ cellule 106 ══════════

    # ============================================================
    # STEP 4.2 - RESNET50 BASELINE
    # Wheat Disease Classification - 17 classes
    # ============================================================




    # ------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------

    SEED = 42
    NUM_CLASSES = len(classes)
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    PATIENCE = 5

    OUTPUT_DIR = os.path.join(
        BASE_DIR, "experiments", "resnet50_baseline"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 65)
    print("RESNET50 BASELINE")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Classes:", NUM_CLASSES)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    # ------------------------------------------------------------
    # 2. MODEL
    # ------------------------------------------------------------

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    # Replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(DEVICE)

    # ------------------------------------------------------------
    # 3. LOSS, OPTIMIZER, SCHEDULER
    # ------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # ------------------------------------------------------------
    # 4. METRICS FUNCTION
    # ------------------------------------------------------------

    def calculate_metrics(y_true, y_pred):

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }


    # ------------------------------------------------------------
    # 5. TRAINING AND VALIDATION
    # ------------------------------------------------------------

    def run_epoch(model, loader, criterion, optimizer=None):

        training = optimizer is not None
        model.train() if training else model.eval()

        total_loss = 0.0
        all_labels = []
        all_predictions = []

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):

                outputs = model(images)
                loss = criterion(outputs, labels)

                if training:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        average_loss = total_loss / len(loader.dataset)

        metrics = calculate_metrics(
            all_labels,
            all_predictions
        )

        metrics["loss"] = average_loss

        return metrics


    # ------------------------------------------------------------
    # 6. TRAINING LOOP WITH EARLY STOPPING
    # ------------------------------------------------------------

    history = []

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "resnet50_best.pth"
    )

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_metrics["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr
        }

        history.append(row)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
        f"LR: {current_lr:.2e} | "
        f"Time: {time.time() - epoch_start:.1f}s"
        )

        # Save best model according to validation macro-F1
        if val_metrics["macro_f1"] > best_val_f1:

            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_f1,
                    "class_to_idx": class_to_idx,
                    "image_size": IMAGE_SIZE
                },
                best_model_path
            )

            print("  -> Best model saved.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # ------------------------------------------------------------
    # 7. SAVE TRAINING HISTORY
    # ------------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"),
        index=False
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print("Best epoch:", best_epoch)
    print("Best validation macro-F1:", round(best_val_f1, 4))
    print("Training time (minutes):", round(elapsed_time / 60, 2))

    # ------------------------------------------------------------
    # 8. TRAINING CURVES
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train"
    )
    axes[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation"
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history_df["epoch"],
        history_df["train_macro_f1"],
        label="Train"
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        label="Validation"
    )
    axes[1].set_title("Training and Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "training_curves.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ------------------------------------------------------------
    # 9. SAVE TRAINING SUMMARY
    # ------------------------------------------------------------

    summary = {
        "architecture": "ResNet50",
        "pretrained_weights": "ImageNet",
        "number_of_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "training_time_seconds": elapsed_time
    }

    with open(
        os.path.join(OUTPUT_DIR, "training_summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nArtifacts saved in:", OUTPUT_DIR)


def stage_14_evaluation():
    """Evaluation sur le test set : depuis checkpoint (114) puis evaluation finale (127).
    
    Origine : cellule(s) 114, 127 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : depend de l'etat du notebook (definis dans cellule 106 : NUM_CLASSES). Executer l'etape correspondante dans la MEME session avec `%run -i`.
    
    ATTENTION : A VALIDER : le choix de cette version/ce role repose sur une deduction (voir CELL_MAP.md).
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v3/experiments/ ; data/processed/classification_v3/test ; outputs/evaluation_resnet50_baseline ; data/processed/
    """
    global test_loader

    # ══════════ cellule 114 ══════════


    # ============================================================
    # 1. Chemins et configuration
    # ============================================================

    MODEL_PATH = PROJECT / (
        "data/processed/classification_v3/experiments/"
    "resnet50_baseline/resnet50_best.pth"
    )
    TEST_DIR = PROJECT / "data/processed/classification_v3/test"

    OUT_DIR = PROJECT / "outputs/evaluation_resnet50_baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    print("Device :", DEVICE)

    # ============================================================
    # 2. Charger le checkpoint
    # ============================================================
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    state_dict = checkpoint["model_state_dict"]
    class_to_idx = checkpoint["class_to_idx"]

    # Vérification du mapping des classes
    idx_to_class = {
        int(index): name for name, index in class_to_idx.items()
    }
    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print("Nombre de classes :", num_classes)
    print("Classes du modèle :", class_names)

    # ============================================================
    # 3. Reconstruire l'architecture
    # ============================================================
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    # ============================================================
    # 4. Préparer le jeu de test
    # IMPORTANT : vérifier que ces transformations correspondent
    # à celles utilisées pendant l'entraînement.
    # ============================================================
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_ds = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=transform
    )

    # Ne pas poursuivre si les labels ne correspondent pas au modèle
    if test_ds.class_to_idx != class_to_idx:
        raise ValueError(
            "Le mapping des classes du test diffère de celui du checkpoint.\n"
        f"Checkpoint : {class_to_idx}\n"
        f"Test       : {test_ds.class_to_idx}"
        )

    if len(test_ds) == 0:
        raise ValueError("Le jeu de test ne contient aucune image.")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Nombre d'images de test :", len(test_ds))

    # ============================================================
    # 5. Prédictions
    # ============================================================
    y_true, y_pred, confidences = [], [], []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            confidence, predictions = probabilities.max(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    # ============================================================
    # 6. Métriques globales et par classe
    # ============================================================
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model": "ResNet50 PyTorch baseline",
        "checkpoint": str(MODEL_PATH),
        "test_samples": int(len(y_true)),
        "number_of_classes": int(num_classes),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "errors": int(np.sum(y_true != y_pred)),
        "correct": int(np.sum(y_true == y_pred))
    }

    print("\n========== RÉSULTATS TEST ==========")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f} ({value * 100:.2f} %)")
        else:
            print(f"{key}: {value}")

    # Sauvegarder les métriques
    with open(OUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame(report).T.to_csv(
        OUT_DIR / "classification_report.csv",
        encoding="utf-8-sig"
    )

    # ============================================================
    # 7. CSV des prédictions
    # ============================================================
    records = []

    for i, (image_path, true_idx) in enumerate(test_ds.samples):
        pred_idx = int(y_pred[i])
        records.append({
            "image_path": str(Path(image_path).relative_to(TEST_DIR)),
            "true_label": idx_to_class[int(true_idx)],
            "predicted_label": idx_to_class[pred_idx],
            "confidence": float(confidences[i]),
            "correct": bool(int(true_idx) == pred_idx)
        })

    predictions_df = pd.DataFrame(records)
    predictions_path = OUT_DIR / "resnet50_test_predictions.csv"
    predictions_df.to_csv(
        predictions_path, index=False, encoding="utf-8-sig"
    )

    # ============================================================
    # 8. Matrice de confusion
    # ============================================================
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Classe prédite",
        ylabel="Classe réelle",
        title="ResNet50 baseline — matrice de confusion"
    )
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 9. Erreurs de classification
    # ============================================================
    errors_df = predictions_df[~predictions_df["correct"]].copy()
    errors_df.to_csv(
        OUT_DIR / "misclassified_images.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFichiers enregistrés dans :", OUT_DIR)
    print("- resnet50_test_predictions.csv")
    print("- test_metrics.json")
    print("- classification_report.csv")
    print("- confusion_matrix.png")
    print("- misclassified_images.csv")

    # ══════════ cellule 127 ══════════

    # ============================================================
    # STEP 4.3 - FINAL TEST EVALUATION
    # ResNet50 - Wheat Disease Classification
    # ============================================================



    # ------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------

    OUTPUT_DIR = (
        "/content/wheat-rust-ai-pipeline/data/processed/"
    "classification_v3/experiments/resnet50_baseline"
    )

    CHECKPOINT_PATH = os.path.join(
        OUTPUT_DIR,
        "resnet50_best.pth"
    )

    assert os.path.isfile(CHECKPOINT_PATH), (
        "Checkpoint not found!"
    )

    # ------------------------------------------------------------
    # 2. LOAD BEST MODEL
    # ------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded.")
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Best validation Macro-F1:",
        checkpoint["best_val_macro_f1"]
    )

    # ------------------------------------------------------------
    # 3. PREDICTIONS ON TEST SET
    # ------------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # ------------------------------------------------------------
    # 4. GLOBAL METRICS
    # ------------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    global_results = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }

    print("\n" + "=" * 60)
    print("GLOBAL TEST RESULTS")
    print("=" * 60)

    for metric, value in global_results.items():
        print(f"{metric:25s}: {value:.4f}")

    # ------------------------------------------------------------
    # 5. CLASSIFICATION REPORT
    # ------------------------------------------------------------

    class_names = [
        idx_to_class[i]
        for i in range(NUM_CLASSES)
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    print("\nCLASSIFICATION REPORT")
    display(report_df.round(4))

    report_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_classification_report.csv"
        )
    )

    # ------------------------------------------------------------
    # 6. CONFUSION MATRIX
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    plt.figure(figsize=(15, 13))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "ResNet50 - Test Confusion Matrix"
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "test_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ------------------------------------------------------------
    # 7. NORMALIZED CONFUSION MATRIX
    # ------------------------------------------------------------

    cm_normalized = (
        cm.astype(float) /
        np.maximum(cm.sum(axis=1, keepdims=True), 1)
    )

    plt.figure(figsize=(15, 13))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "ResNet50 - Normalized Test Confusion Matrix"
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "test_normalized_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ------------------------------------------------------------
    # 8. SAVE GLOBAL RESULTS
    # ------------------------------------------------------------

    with open(
        os.path.join(
            OUTPUT_DIR,
            "test_global_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            global_results,
            f,
            indent=4
        )

    # ------------------------------------------------------------
    # 9. SAVE PREDICTIONS
    # ------------------------------------------------------------

    predictions_df = pd.DataFrame({
        "true_label": [
            idx_to_class[i] for i in y_true
        ],
        "predicted_label": [
            idx_to_class[i] for i in y_pred
        ],
        "correct": y_true == y_pred,
        "confidence": y_prob.max(axis=1)
    })

    predictions_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions.csv"
        ),
        index=False
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETED")
    print("=" * 60)

    print("Test images:", len(y_true))
    print("Results directory:", OUTPUT_DIR)


def stage_15_error_analysis():
    """Analyse des erreurs du ResNet50 baseline.
    
    Origine : cellule(s) 128 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/processed/
    """

    # ══════════ cellule 128 ══════════

    # ============================================================
    # STEP 4.4 - ERROR ANALYSIS
    # ResNet50 - Test set
    # ============================================================


    OUTPUT_DIR = (
        "/content/wheat-rust-ai-pipeline/data/processed/"
    "classification_v3/experiments/resnet50_baseline"
    )

    DATASET_DIR = (
        "/content/wheat-rust-ai-pipeline/data/processed/"
    "classification_v3"
    )

    # ------------------------------------------------------------
    # 1. LOAD PREDICTIONS AND MANIFEST
    # ------------------------------------------------------------

    predictions = pd.read_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv")
    )

    manifest = pd.read_csv(
        os.path.join(DATASET_DIR, "classification_manifest.csv")
    )

    # Use destination_path if available; otherwise image_path.
    if "destination_path" in manifest.columns:
        path_column = "destination_path"
    elif "image_path" in manifest.columns:
        path_column = "image_path"
    else:
        raise ValueError("No image path column found.")

    test_manifest = manifest[
        manifest["new_split"] == "test"
    ].copy()

    test_manifest = test_manifest.reset_index(drop=True)

    # Confirm the order used by the test DataLoader.
    assert len(test_manifest) == len(predictions), (
        "Prediction/manifest row counts differ."
    )

    # Check the class order against the predictions.
    assert (
        test_manifest["canonical_class"].values
        == predictions["true_label"].values
    ).all(), (
        "Prediction order does not match manifest order. "
    "Check test DataLoader ordering."
    )

    predictions["image_path"] = test_manifest[
        path_column
    ].values

    # ------------------------------------------------------------
    # 2. IDENTIFY ERRORS
    # ------------------------------------------------------------

    errors = predictions[
        predictions["correct"] == False
    ].copy()

    errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "misclassified_test_images.csv"
        ),
        index=False
    )

    print("Total test images:", len(predictions))
    print("Correct predictions:", predictions["correct"].sum())
    print("Misclassified images:", len(errors))
    print(
        "Error rate: {:.2f}%".format(
            100 * len(errors) / len(predictions)
        )
    )

    # ------------------------------------------------------------
    # 3. MOST FREQUENT CONFUSIONS
    # ------------------------------------------------------------

    confusions = (
        errors.groupby(
            ["true_label", "predicted_label"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("\nTOP 20 CONFUSIONS")
    display(confusions.head(20))

    confusions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_class_confusions.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------
    # 4. ERROR RATE BY TRUE CLASS
    # ------------------------------------------------------------

    class_errors = (
        errors.groupby("true_label")
        .size()
        .rename("errors")
    )

    class_support = (
        predictions.groupby("true_label")
        .size()
        .rename("support")
    )

    error_analysis = pd.concat(
        [class_errors, class_support],
        axis=1
    ).fillna(0)

    error_analysis["errors"] = (
        error_analysis["errors"].astype(int)
    )

    error_analysis["error_rate"] = (
        error_analysis["errors"] /
        error_analysis["support"]
    )

    error_analysis = error_analysis.sort_values(
        "error_rate",
        ascending=False
    )

    print("\nERROR RATE BY CLASS")
    display(error_analysis.round(4))

    error_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "error_rate_by_class.csv"
        )
    )

    # ------------------------------------------------------------
    # 5. VISUALIZE MOST CONFUSED IMAGES
    # ------------------------------------------------------------

    def show_errors(
        dataframe,
        true_class=None,
        max_images=12
    ):

        subset = dataframe.copy()

        if true_class is not None:
            subset = subset[
                subset["true_label"] == true_class
            ]

        subset = subset.head(max_images)

        if len(subset) == 0:
            print("No images found.")
            return

        cols = 4
        rows = (len(subset) + cols - 1) // cols

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(16, 4 * rows)
        )

        axes = axes.flatten()

        for ax, (_, row) in zip(axes, subset.iterrows()):

            try:
                img = Image.open(
                    row["image_path"]
                ).convert("RGB")

                ax.imshow(img)

                ax.set_title(
                    f"True: {row['true_label']}\n"
                f"Pred: {row['predicted_label']}\n"
                f"Confidence: {row['confidence']:.3f}",
                    fontsize=9
                )

            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    str(e),
                    ha="center",
                    va="center"
                )

            ax.axis("off")

        for ax in axes[len(subset):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


    # ------------------------------------------------------------
    # 6. DISPLAY ERRORS FOR IMPORTANT CLASSES
    # ------------------------------------------------------------

    for class_name in [
        "Tan_Spot",
        "Leaf_Blight",
        "Mite",
        "Black_Rust"
    ]:

        print("\n" + "=" * 60)
        print("ERROR ANALYSIS:", class_name)
        print("=" * 60)

        show_errors(
            errors,
            true_class=class_name,
            max_images=12
        )

    print("\nError analysis completed.")


def stage_16_source_wise_evaluation():
    """Evaluation par source d'origine des images (requiert test_loader de step14).
    
    Origine : cellule(s) 132 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : depend de l'etat du notebook (definis dans cellule 114 : test_loader). Executer l'etape correspondante dans la MEME session avec `%run -i`.
    
    ATTENTION : A VALIDER : le choix de cette version/ce role repose sur une deduction (voir CELL_MAP.md).
    ATTENTION : Derniere execution enregistree en ERREUR : cellule 132 (ValueError)
    
    Chemins references : data/processed/classification_v3
    """

    # ══════════ cellule 132 ══════════

    # ============================================================
    # SOURCE-WISE EVALUATION — RESNET50
    # Classification_v3 | Wheat Disease Benchmark
    # ============================================================



    # ------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------

    BASE_DIR = "/content/wheat-rust-ai-pipeline/data/processed/classification_v3"

    MANIFEST_PATH = os.path.join(
        BASE_DIR, "classification_manifest.csv"
    )

    EXPERIMENT_DIR = os.path.join(
        BASE_DIR, "experiments/resnet50_baseline"
    )

    PREDICTIONS_PATH = os.path.join(
        EXPERIMENT_DIR, "test_predictions.csv"
    )

    OUTPUT_DIR = os.path.join(
        EXPERIMENT_DIR, "source_wise_evaluation"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 2. VERIFICATION DES FICHIERS ET DU DATALOADER
    # ------------------------------------------------------------

    assert os.path.exists(MANIFEST_PATH), "Manifest introuvable"
    assert os.path.exists(PREDICTIONS_PATH), "Predictions introuvables"

    assert "test_loader" in globals(), (
        "test_loader absent : réexécutez la cellule de création des DataLoaders."
    )

    manifest = pd.read_csv(MANIFEST_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_manifest = {"dataset", "filename"}
    missing = required_manifest - set(manifest.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes dans le manifest : {missing}")

    print("Manifest :", manifest.shape)
    print("Predictions :", predictions.shape)

    # ------------------------------------------------------------
    # 3. RECUPERATION DE L'ORDRE EXACT DU TEST DATASET
    # ------------------------------------------------------------

    test_dataset = test_loader.dataset

    if not hasattr(test_dataset, "samples"):
        raise ValueError(
            "Le test_dataset ne possède pas l'attribut samples. "
        "Il faut récupérer les chemins dans le Dataset utilisé."
        )

    samples = test_dataset.samples

    test_paths = [sample[0] for sample in samples]
    test_numeric_labels = [sample[1] for sample in samples]

    class_names = test_dataset.classes

    if len(test_paths) != len(predictions):
        raise ValueError(
            f"Nombre d'images ({len(test_paths)}) différent "
        f"du nombre de prédictions ({len(predictions)})."
        )

    if not {"true_label", "predicted_label"}.issubset(predictions.columns):
        raise ValueError("Colonnes true_label/predicted_label absentes.")

    # Vérifier que l'ordre du CSV correspond à celui du Dataset.
    expected_true_labels = [
        class_names[int(label)] for label in test_numeric_labels
    ]

    csv_true_labels = predictions["true_label"].astype(str).tolist()

    if expected_true_labels != csv_true_labels:
        raise ValueError(
            "Désalignement détecté entre test_loader et test_predictions.csv. "
        "Ne poursuivez pas avant d'avoir reconstruit les prédictions "
        "dans le même ordre que le test Dataset."
        )

    print("SUCCESS : ordre des prédictions vérifié.")

    # ------------------------------------------------------------
    # 4. ASSOCIATION IMAGE -> DATASET SOURCE
    # ------------------------------------------------------------

    manifest = manifest.copy()

    manifest["filename"] = manifest["filename"].astype(str).apply(
        lambda x: os.path.basename(x)
    )

    # Une image doit correspondre à une seule source.
    source_counts = manifest.groupby("filename")["dataset"].nunique()

    ambiguous = source_counts[source_counts > 1]

    if len(ambiguous) > 0:
        raise ValueError(
            "Certains noms de fichiers correspondent à plusieurs sources. "
        "Utilisez un identifiant SHA-256 pour établir la correspondance."
        )

    filename_to_source = (
        manifest.drop_duplicates("filename")
        .set_index("filename")["dataset"]
        .to_dict()
    )

    test_filenames = [os.path.basename(p) for p in test_paths]

    sources = [
        filename_to_source.get(filename)
        for filename in test_filenames
    ]

    missing_sources = sum(source is None for source in sources)

    if missing_sources > 0:
        missing_names = [
            test_filenames[i]
            for i, source in enumerate(sources)
            if source is None
        ]

        pd.DataFrame({"filename": missing_names}).to_csv(
            os.path.join(OUTPUT_DIR, "unmatched_test_images.csv"),
            index=False
        )

        raise ValueError(
            f"{missing_sources} images sans source identifiée. "
        "Consultez unmatched_test_images.csv."
        )

    # ------------------------------------------------------------
    # 5. CONSTRUCTION DU DATAFRAME FINAL
    # ------------------------------------------------------------

    df = predictions.copy()

    df["image_path"] = test_paths
    df["filename"] = test_filenames
    df["dataset"] = sources

    df.to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions_with_source.csv"),
        index=False
    )

    print("\nImages associées aux sources :", len(df))
    print("\nDistribution des images par source :")
    print(df["dataset"].value_counts())

    # ------------------------------------------------------------
    # 6. METRIQUES GLOBALES PAR SOURCE
    # ------------------------------------------------------------

    source_results = []

    for source, group in df.groupby("dataset"):

        y_true = group["true_label"].astype(str)
        y_pred = group["predicted_label"].astype(str)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_names,
            average="macro",
            zero_division=0
        )

        source_results.append({
            "dataset": source,
            "support": len(group),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "errors": int((y_true != y_pred).sum()),
            "error_rate": float((y_true != y_pred).mean())
        })

    source_metrics = pd.DataFrame(source_results)

    source_metrics = source_metrics.sort_values(
        "macro_f1", ascending=False
    )

    source_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "source_level_metrics.csv"),
        index=False
    )

    print("\n" + "=" * 70)
    print("SOURCE-WISE GLOBAL METRICS")
    print("=" * 70)
    print(source_metrics.to_string(index=False))

    # ------------------------------------------------------------
    # 7. METRIQUES PAR SOURCE ET PAR CLASSE
    # ------------------------------------------------------------

    class_source_results = []

    for source, group in df.groupby("dataset"):

        report = classification_report(
            group["true_label"],
            group["predicted_label"],
            labels=class_names,
            output_dict=True,
            zero_division=0
        )

        for class_name in class_names:

            values = report[class_name]

            class_source_results.append({
                "dataset": source,
                "class": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"])
            })

    class_source_metrics = pd.DataFrame(class_source_results)

    class_source_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "source_class_metrics.csv"),
        index=False
    )

    # ------------------------------------------------------------
    # 8. MATRICES DE CONFUSION PAR SOURCE
    # ------------------------------------------------------------

    confusion_dir = os.path.join(OUTPUT_DIR, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)

    for source, group in df.groupby("dataset"):

        cm = confusion_matrix(
            group["true_label"],
            group["predicted_label"],
            labels=class_names
        )

        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in str(source)
        )

        pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names
        ).to_csv(
            os.path.join(confusion_dir, f"{safe_name}_confusion_matrix.csv")
        )

    # ------------------------------------------------------------
    # 9. DISTRIBUTION DES CLASSES PAR SOURCE
    # ------------------------------------------------------------

    source_class_distribution = pd.crosstab(
        df["dataset"],
        df["true_label"]
    )

    source_class_distribution.to_csv(
        os.path.join(OUTPUT_DIR, "source_class_distribution.csv")
    )

    # ------------------------------------------------------------
    # 10. RESUME JSON
    # ------------------------------------------------------------

    summary = {
        "total_test_images": int(len(df)),
        "number_of_sources": int(df["dataset"].nunique()),
        "global_accuracy": float(accuracy_score(
            df["true_label"], df["predicted_label"]
        )),
        "global_error_count": int(
            (df["true_label"] != df["predicted_label"]).sum()
        ),
        "sources": source_metrics.to_dict(orient="records")
    }

    with open(
        os.path.join(OUTPUT_DIR, "source_wise_summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    # ------------------------------------------------------------
    # 11. VISUALISATION
    # ------------------------------------------------------------


    plot_df = source_metrics.sort_values("macro_f1")

    plt.figure(figsize=(11, max(5, len(plot_df) * 0.55)))

    plt.barh(
        plot_df["dataset"].astype(str),
        plot_df["macro_f1"]
    )

    plt.xlabel("Macro-F1")
    plt.ylabel("Source dataset")
    plt.title("ResNet50 — Macro-F1 by Dataset Source")
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "macro_f1_by_source.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ------------------------------------------------------------
    # 12. FIN
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCE-WISE EVALUATION COMPLETED")
    print("=" * 70)
    print("Output directory:", OUTPUT_DIR)
    print("\nGenerated files:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for filename in files:
            print(os.path.relpath(os.path.join(root, filename), OUTPUT_DIR))


def stage_17_predict_with_paths():
    """Inference depuis le checkpoint avec chemins conserves -> test_predictions_with_paths.csv.
    
    Origine : cellule(s) 158 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/drive/MyDrive/Colab Notebooks/ ; data/processed/
    """

    # ══════════ cellule 158 ══════════



    # --------------------------------------------------
    # 1. Chemins
    # --------------------------------------------------
    checkpoint_path = (
        "/content/drive/MyDrive/Colab Notebooks/"
    "WHEAT_RUST_RESEARCH1/03_EXPERIMENTS/"
    "RECOVERED_RESNET50_BASELINE/resnet50_best.pth"
    )

    test_dir = (
        "/content/wheat-rust-ai-pipeline/data/processed/"
    "classification_v3/test"
    )

    output_csv = (
        "/content/drive/MyDrive/Colab Notebooks/"
    "WHEAT_RUST_RESEARCH1/03_EXPERIMENTS/"
    "RECOVERED_RESNET50_BASELINE/test_predictions_with_paths.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    # --------------------------------------------------
    # 2. Charger le checkpoint
    # --------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    image_size = checkpoint.get("image_size", 224)

    print("Classes :", class_to_idx)
    print("Nombre de classes :", num_classes)
    print("Taille d'image :", image_size)

    # --------------------------------------------------
    # 3. Prétraitement
    # IMPORTANT : utiliser les mêmes transformations
    # qu'au test initial si elles sont documentées.
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ImageFolder fournit les chemins et les étiquettes.
    dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    # Vérifier l'ordre des classes par rapport au checkpoint.
    print("Classes ImageFolder :", dataset.classes)

    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Les indices des classes ne correspondent pas au checkpoint.\n"
        f"Checkpoint : {class_to_idx}\n"
        f"ImageFolder : {dataset.class_to_idx}"
        )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------
    # 4. Reconstruire ResNet50 et charger les poids
    # --------------------------------------------------
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --------------------------------------------------
    # 5. Inférence avec conservation des chemins
    # --------------------------------------------------
    rows = []
    offset = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            predictions = predictions.cpu().numpy()
            confidences = confidences.cpu().numpy()
            labels = labels.numpy()

            batch_paths = [
                dataset.samples[i][0]
                for i in range(offset, offset + len(labels))
            ]

            for path, true_idx, pred_idx, confidence in zip(
                batch_paths, labels, predictions, confidences
            ):
                true_label = idx_to_class[int(true_idx)]
                predicted_label = idx_to_class[int(pred_idx)]

                rows.append({
                    "image_path": path,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": true_label == predicted_label,
                    "confidence": float(confidence)
                })

            offset += len(labels)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\nCSV enregistré :", output_csv)
    print("Nombre d'images :", len(df))
    print("Nombre d'erreurs :", int((~df["correct"]).sum()))
    print("Accuracy recalculée :", round(df["correct"].mean() * 100, 2), "%")
    print("\nExemples d'erreurs :")
    print(df.loc[~df["correct"]].head(10).to_string(index=False))


STAGES = {
    "13": stage_13_dataset_and_training,
    "14": stage_14_evaluation,
    "15": stage_15_error_analysis,
    "16": stage_16_source_wise_evaluation,
    "17": stage_17_predict_with_paths,
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
