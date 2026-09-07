"""
pipelines/attention_position_ablation.py
------------------------------------------
Ablation de POSITION pour LES QUATRE mécanismes d'attention (SE, CBAM,
Triplet Attention, TAM) — pas seulement TAM.

Contexte :
    pipelines/tam_position_ablation.py a montré comment insérer TAM à 4
    profondeurs différentes du ResNet50 via chirurgie de graphe
    fonctionnelle (réutilisation des mêmes couches -> poids ImageNet
    partagés, jamais copiés/réinitialisés). Pour une comparaison
    RIGOUREUSE entre mécanismes d'attention, il faut appliquer EXACTEMENT
    la même procédure à SE, CBAM et Triplet Attention — sinon on
    compare "TAM testé à 4 positions" à "SE/CBAM/Triplet testés à 1
    seule position", ce qui biaise la comparaison en faveur ou en
    défaveur de TAM selon la position où il a été le plus chanceux.

Ce module réutilise directement `_insert_layer_after` de
pipelines/tam_position_ablation.py (fonction générique, ne dépend pas de
TAM) et l'applique à SEBlock, CBAMBlock, TripletAttention (définis dans
pipelines/attention_modules.py).

Grille complète couverte, une fois ce module enregistré :

                stage2      stage3      stage4      stage5 (final)
    SE          nouveau     nouveau     nouveau     resnet50_se (existant)
    CBAM        nouveau     nouveau     nouveau     resnet50_cbam (existant)
    Triplet     nouveau     nouveau     nouveau     resnet50_triplet (existant)
    TAM         resnet50_tam_stage2/3/4/5 (déjà fait, tam_position_ablation.py)

Usage :
    from pipelines.attention_position_ablation import build_resnet50_attention_at
    model = build_resnet50_attention_at(
        input_shape=(256, 256, 3), num_classes=17,
        stage="stage3", attention_type="cbam",
    )
"""

import functools

from tensorflow import keras
from tensorflow.keras import layers

from pipelines.tam_position_ablation import _insert_layer_after, STAGE_LAYER_NAMES
from pipelines.attention_modules import SEBlock, CBAMBlock, TripletAttention


# Constructeurs de couche d'attention, par nom -- même mécanisme, même
# hyperparamètres par défaut que dans attention_modules.py (reduction_ratio=16,
# spatial_kernel_size=7, kernel_size=7), pour une comparaison qui ne fait
# varier QUE la position, rien d'autre.
_ATTENTION_LAYER_FACTORIES = {
    "se": lambda stage: SEBlock(reduction_ratio=16, name=f"se_{stage}"),
    "cbam": lambda stage: CBAMBlock(reduction_ratio=16, spatial_kernel_size=7,
                                     name=f"cbam_{stage}"),
    "triplet": lambda stage: TripletAttention(kernel_size=7, name=f"triplet_{stage}"),
}


def build_resnet50_attention_at(input_shape: tuple, num_classes: int, stage: str,
                                 attention_type: str,
                                 freeze_base: bool = True) -> keras.Model:
    """
    ResNet50 + {SE, CBAM, Triplet} inséré à une profondeur choisie.

    Args:
        stage : une des clés de STAGE_LAYER_NAMES
                 ("stage2", "stage3", "stage4", "stage5").
                 "stage5" reproduit la position des builders existants
                 (build_resnet50_se/cbam/triplet dans attention_modules.py)
                 -- utile comme témoin de cohérence entre implémentations.
        attention_type : "se", "cbam", ou "triplet".

    Le reste de l'architecture (preprocessing, tête de classification)
    est identique aux builders existants, pour une comparaison contrôlée
    ne faisant varier QUE la position d'insertion.
    """
    if stage not in STAGE_LAYER_NAMES:
        raise ValueError(f"stage doit être dans {list(STAGE_LAYER_NAMES)}, reçu '{stage}'")
    if attention_type not in _ATTENTION_LAYER_FACTORIES:
        raise ValueError(
            f"attention_type doit être dans {list(_ATTENTION_LAYER_FACTORIES)}, "
            f"reçu '{attention_type}'"
        )

    base_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    target_layer_name = STAGE_LAYER_NAMES[stage]
    layer_factory = _ATTENTION_LAYER_FACTORIES[attention_type]
    base_model_with_attn = _insert_layer_after(
        base_model, target_layer_name,
        new_layer_factory=lambda: layer_factory(stage),
    )
    base_model_with_attn.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    features = base_model_with_attn(x, training=False)

    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"ResNet50_{attention_type.upper()}_{stage}")
    model.base_model = base_model_with_attn
    return model


def register_attention_position_ablation_models(model_builders: dict) -> None:
    """
    Ajoute resnet50_{se,cbam,triplet}_stage{2,3,4} au dict MODEL_BUILDERS.
    stage5 n'est PAS ajouté ici : resnet50_se/cbam/triplet (position finale)
    existent déjà dans pipelines/attention_modules.py -- pas de doublon.
    """
    for attention_type in _ATTENTION_LAYER_FACTORIES:
        for stage in ("stage2", "stage3", "stage4"):
            name = f"resnet50_{attention_type}_{stage}"
            model_builders[name] = functools.partial(
                build_resnet50_attention_at, stage=stage, attention_type=attention_type
            )