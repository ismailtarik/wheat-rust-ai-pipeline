"""
---------------------
Phase 2 — Étape 1 : Architectures de classification.

Modèles disponibles :
  - cnn_custom      : CNN simple construit from scratch (baseline de référence)
  - resnet50         : ResNet50 pré-entraîné ImageNet (transfer learning)
  - efficientnetb0   : EfficientNetB0 pré-entraîné ImageNet (transfer learning)

Usage :
    from pipelines.models import build_model
    model = build_model("resnet50", input_shape=(256,256,3), num_classes=15)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_custom(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    CNN simple construit from scratch.
    Sert de baseline de référence (sans transfer learning) pour mesurer
    l'apport réel des modèles pré-entraînés.

    Architecture :
      4 blocs Conv2D + BatchNorm + MaxPool, puis GlobalAveragePooling
      et 2 couches denses avec Dropout.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="CNN_Custom")
    return model


def build_resnet50(input_shape: tuple, num_classes: int,
                    freeze_base: bool = True) -> keras.Model:
    """
    ResNet50 pré-entraîné sur ImageNet, adapté à la classification
    des maladies du blé via transfer learning.

    Args:
        freeze_base : si True, gèle les poids du backbone ResNet50
                      (entraînement uniquement de la tête de classification).
                      À passer à False lors du fine-tuning.
    """
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    # ResNet50 attend des images préprocessées façon ImageNet (pas juste [0,1])
    x = layers.Rescaling(255.0)(inputs)  # on annule la normalisation [0,1] du pipeline
    x = keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ResNet50_TransferLearning")
    model.base_model = base_model   # référence pour le fine-tuning ultérieur
    return model


def build_efficientnetb0(input_shape: tuple, num_classes: int,
                          freeze_base: bool = True) -> keras.Model:
    """
    EfficientNetB0 pré-entraîné sur ImageNet, adapté à la classification
    des maladies du blé via transfer learning.

    Args:
        freeze_base : si True, gèle les poids du backbone EfficientNetB0.
    """
    base_model = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    # EfficientNet attend des pixels en [0, 255] (normalisation interne)
    x = layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_TransferLearning")
    model.base_model = base_model
    return model


# Registre des architectures disponibles
MODEL_BUILDERS = {
    "cnn_custom"    : build_cnn_custom,
    "resnet50"      : build_resnet50,
    "efficientnetb0": build_efficientnetb0,
}


def build_model(model_name: str, input_shape: tuple, num_classes: int,
                 freeze_base: bool = True) -> keras.Model:
    """
    Factory function — construit le modèle demandé par son nom.

    Args:
        model_name  : "cnn_custom" | "resnet50" | "efficientnetb0"
        input_shape : (H, W, C)
        num_classes : nombre de classes en sortie
        freeze_base : pour les modèles de transfer learning, gèle le backbone

    Returns:
        Modèle Keras non compilé
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"❌ Modèle inconnu : '{model_name}'. "
            f"Choix possibles : {list(MODEL_BUILDERS.keys())}"
        )

    builder = MODEL_BUILDERS[model_name]

    if model_name == "cnn_custom":
        return builder(input_shape, num_classes)
    else:
        return builder(input_shape, num_classes, freeze_base=freeze_base)


def unfreeze_for_finetuning(model: keras.Model, num_layers_to_unfreeze: int = 30) -> None:
    """
    Dégèle les N dernières couches du backbone pour le fine-tuning.
    Ne s'applique qu'aux modèles de transfer learning (ResNet50, EfficientNetB0).

    Args:
        model                   : modèle construit par build_model()
        num_layers_to_unfreeze  : nombre de couches à dégeler depuis la fin du backbone
    """
    if not hasattr(model, "base_model"):
        print(f"  ⚠️  Le modèle '{model.name}' n'a pas de backbone à dégeler (CNN custom).")
        return

    base_model = model.base_model
    base_model.trainable = True

    # Gèle toutes les couches sauf les N dernières
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  🔓 Fine-tuning activé : {n_trainable}/{len(base_model.layers)} "
          f"couches du backbone dégelées.")