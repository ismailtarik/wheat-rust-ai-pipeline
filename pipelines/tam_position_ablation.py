"""
pipelines/tam_position_ablation.py
---------------------------------
Ablation de POSITION du Texture Attention Module (TAM) dans ResNet50.

Contexte scientifique :
    L'implémentation actuelle (pipelines/texture_attention.py) insère TAM
    à un seul point fixe : la feature map finale du backbone (B, 8, 8, 2048).
    L'audit du benchmark a soulevé une question légitime (cf.
    FINAL_AUDIT_REPORT) : à cette profondeur, la feature map est déjà très
    abstraite/sémantique et de résolution spatiale très grossière (8×8) —
    potentiellement trop tard pour qu'un filtre de Gabor (conçu pour
    détecter des motifs de texture LOCAUX) soit réellement utile.

    Ce module NE MODIFIE PAS l'implémentation existante de TAM
    (pipelines/texture_attention.py, insertion finale) — il ajoute des
    variantes alternatives, à des profondeurs différentes, pour une
    étude d'ablation contrôlée. build_resnet50_tam (profondeur finale,
    existante) reste le point de comparaison E4 principal.

Points d'insertion disponibles (ResNet50, sorties de fin de stage,
vérifiées : couches Activation, 0 poids propre, donc aucun risque de
perturber les poids ImageNet pré-entraînés du reste du réseau) :

    stage2  -> conv2_block3_out   (B, 64, 64, 256)   -- le plus fin, le plus "textural"
    stage3  -> conv3_block4_out   (B, 32, 32, 512)
    stage4  -> conv4_block6_out   (B, 16, 16, 1024)
    stage5  -> conv5_block3_out   (B, 8,  8,  2048)  -- position actuelle de TAM (E4)

Méthode d'insertion (chirurgie de graphe fonctionnelle Keras) :
    On ne peut pas simplement "couper" un keras.Model au milieu et
    reprendre le calcul, à cause des connexions résiduelles (Add) de
    ResNet50 qui référencent plusieurs tenseurs intermédiaires. La
    technique standard consiste à REJOUER chaque couche du modèle
    d'origine, dans l'ordre topologique, sur de NOUVEAUX tenseurs, en
    insérant TAM juste après la couche cible. Comme on réutilise les
    MÊMES objets Layer (pas de copie), les poids ImageNet pré-entraînés
    sont automatiquement partagés — aucune réinitialisation, aucune
    perte de pretrained weights.

Usage :
    from pipelines.tam_position_ablation import build_resnet50_tam_at
    model = build_resnet50_tam_at(
        input_shape=(256, 256, 3), num_classes=17, stage="stage3"
    )
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pipelines.texture_attention import TextureAttentionModule


# Nom de la couche Activation Keras marquant la fin de chaque stage
# ResNet50, et forme de sortie associée pour un input 256x256 (pour
# documentation — la fonction reste générique à toute résolution).
STAGE_LAYER_NAMES = {
    "stage2": "conv2_block3_out",   # (B, H/4,  W/4,  256)
    "stage3": "conv3_block4_out",   # (B, H/8,  W/8,  512)
    "stage4": "conv4_block6_out",   # (B, H/16, W/16, 1024)
    "stage5": "conv5_block3_out",   # (B, H/32, W/32, 2048) -- position actuelle (E4)
}


def _insert_layer_after(base_model: keras.Model, target_layer_name: str,
                         new_layer_factory) -> keras.Model:
    """
    Rejoue le graphe fonctionnel de `base_model` couche par couche, en
    insérant `new_layer_factory()` juste après la couche nommée
    `target_layer_name`. Réutilise les MÊMES objets Layer (donc les
    mêmes poids pré-entraînés) — seule la connectivité du graphe change.

    Fonctionne pour tout modèle Functional Keras dont `.layers` est en
    ordre topologique (garanti pour les modèles construits via
    `keras.applications.*`, y compris ceux avec connexions résiduelles).
    """
    if target_layer_name not in [l.name for l in base_model.layers]:
        raise ValueError(
            f"Couche '{target_layer_name}' introuvable dans le modèle "
            f"(base_model = {base_model.name}). Vérifiez STAGE_LAYER_NAMES."
        )

    # tensor_map : associe chaque tenseur de sortie ORIGINAL (par id) au
    # nouveau tenseur correspondant dans le graphe reconstruit.
    tensor_map = {}
    tensor_map[id(base_model.input)] = base_model.input

    inserted = False
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        orig_inputs = layer.input
        if isinstance(orig_inputs, list):
            new_inputs = [tensor_map[id(t)] for t in orig_inputs]
        else:
            new_inputs = tensor_map[id(orig_inputs)]

        # Rejoue la couche d'origine (même objet -> mêmes poids partagés)
        new_output = layer(new_inputs)

        if layer.name == target_layer_name:
            new_output = new_layer_factory()(new_output)
            inserted = True

        tensor_map[id(layer.output)] = new_output

    if not inserted:
        # Ne devrait jamais arriver vu la vérification en début de fonction,
        # gardé par sécurité pour ne jamais retourner silencieusement un
        # modèle SANS l'insertion demandée.
        raise RuntimeError(
            f"Insertion de la couche après '{target_layer_name}' a échoué "
            f"silencieusement — ne pas utiliser ce modèle."
        )

    final_output = tensor_map[id(base_model.output)]
    return keras.Model(base_model.input, final_output,
                        name=f"{base_model.name}_TAM_at_{target_layer_name}")


def build_resnet50_tam_at(input_shape: tuple, num_classes: int, stage: str,
                           freeze_base: bool = True,
                           n_orientations: int = 8) -> keras.Model:
    """
    ResNet50 + TAM inséré à une profondeur choisie (ablation de position).

    Args:
        stage : une des clés de STAGE_LAYER_NAMES
                 ("stage2", "stage3", "stage4", "stage5").
                 "stage5" reproduit exactement la position de
                 build_resnet50_tam (E4 existant) — utile comme témoin
                 de cohérence entre les deux implémentations.

    Le reste de l'architecture (préprocessing, tête de classification)
    est identique à build_resnet50_tam, pour une comparaison contrôlée
    ne faisant varier QUE la position d'insertion.
    """
    if stage not in STAGE_LAYER_NAMES:
        raise ValueError(f"stage doit être dans {list(STAGE_LAYER_NAMES)}, reçu '{stage}'")

    base_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    target_layer_name = STAGE_LAYER_NAMES[stage]
    base_model_with_tam = _insert_layer_after(
        base_model, target_layer_name,
        new_layer_factory=lambda: TextureAttentionModule(
            n_orientations=n_orientations, name=f"tam_{stage}"
        ),
    )
    base_model_with_tam.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    features = base_model_with_tam(x, training=False)   # (H/32, W/32, 2048)

    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"ResNet50_TAM_{stage}")
    model.base_model = base_model_with_tam
    return model


# Enregistrement paresseux dans le registre central de modèles, un nom
# par position testée -- E4a..E4d dans le protocole d'ablation.
def register_position_ablation_models(model_builders: dict) -> None:
    """
    Ajoute resnet50_tam_stage2 / stage3 / stage4 / stage5 au dict
    MODEL_BUILDERS de pipelines/models.py. Appelé depuis models.py de
    façon paresseuse, comme les autres enregistrements TAM/attention.
    """
    import functools
    for stage in STAGE_LAYER_NAMES:
        model_builders[f"resnet50_tam_{stage}"] = functools.partial(
            build_resnet50_tam_at, stage=stage
        )