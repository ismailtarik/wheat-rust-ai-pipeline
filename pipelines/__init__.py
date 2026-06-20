"""
Package pipelines.

Phase 1 — Dataset Acquisition & Preprocessing :
    data_collection  → Étape 1 : Téléchargement (Kaggle API)
    validation       → Étape 2 : Filtrage classes + EDA
    split_dataset    → Étape 3 : Split stratifié Train/Val/Test + exports CSV
    preprocessing    → Étape 4 : Pipelines tf.data (resize, normalisation, augmentation)

Phase 2 — Modèles Baseline :
    models               → Architectures (CNN custom, ResNet50, EfficientNetB0)
    train_classification → Entraînement + évaluation + comparaison des modèles
"""

from .data_collection import collect_data
from .validation      import validate_dataset
from .split_dataset   import split_dataset
from .preprocessing   import build_tf_datasets
from .models           import build_model, unfreeze_for_finetuning
from .train_classification import run_classification_phase

__all__ = [
    "collect_data",
    "validate_dataset",
    "split_dataset",
    "build_tf_datasets",
    "build_model",
    "unfreeze_for_finetuning",
    "run_classification_phase",
]