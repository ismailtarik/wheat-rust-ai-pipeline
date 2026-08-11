"""
pipelines/texture_attention.py
---------------------------------
Contribution 1 — Texture Attention Module (TAM).

Motivation scientifique :
    Les maladies du blé présentent des signatures texturales distinctives
    (pustules de rouille, revêtement poudreux du mildiou, masses granuleuses
    du smut, lésions elliptiques du blast). Les mécanismes d'attention
    génériques (SE-block, CBAM) pondèrent les canaux/l'espace sans exploiter
    explicitement cette information texturale. TAM introduit un mécanisme
    d'attention guidé par une banque de filtres de Gabor, calculant une carte
    d'énergie texturale locale utilisée pour moduler les feature maps du
    backbone via un gating résiduel.

Architecture du module (pour une feature map X de forme H×W×C) :
    1. Banque de filtres de Gabor à K orientations (0°, 22.5°, 45°, ..., 157.5°)
       appliqués en depthwise convolution (poids fixes, non entraînables —
       garantit une réponse texturale interprétable et stable)
    2. Magnitude de réponse par orientation, agrégée (max) sur les K
       orientations → carte d'énergie texturale T (H×W×C)
    3. Compression canal (moyenne sur C) → T' (H×W×1)
    4. Attention spatiale : conv 1×1 + sigmoid → masque A (H×W×1)
    5. Sortie : X' = X + X ⊙ A (gating résiduel — préserve le signal
       original tout en amplifiant les régions à forte énergie texturale)

Usage :
    from pipelines.texture_attention import TextureAttentionModule, build_resnet50_tam
    model = build_resnet50_tam(input_shape=(256,256,3), num_classes=17)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────
# Banque de filtres de Gabor (générés analytiquement, poids fixes)
# ─────────────────────────────────────────────────────────────

def _make_gabor_kernel(ksize: int, sigma: float, theta: float,
                        lambd: float, gamma: float, psi: float = 0.0) -> np.ndarray:
    """
    Génère un noyau de Gabor 2D réel (partie cosinus) — filtre orienté
    classique pour l'analyse de texture directionnelle.

    Args:
        ksize : taille du noyau (impair, ex: 7)
        sigma : écart-type de l'enveloppe gaussienne
        theta : orientation du filtre (radians)
        lambd : longueur d'onde de la sinusoïde
        gamma : ratio d'aspect spatial
        psi   : décalage de phase
    """
    half = ksize // 2
    y, x = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1), indexing="ij")

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
    gb *= np.cos(2 * np.pi * x_theta / lambd + psi)
    return gb.astype(np.float32)


def _build_gabor_bank(ksize: int = 7, n_orientations: int = 8) -> np.ndarray:
    """
    Construit une banque de n_orientations filtres de Gabor couvrant
    0° à 157.5° (pas de 180°/n_orientations).

    Returns:
        np.ndarray de forme (ksize, ksize, 1, n_orientations) — prête à
        être utilisée comme poids d'une couche DepthwiseConv2D répétée
        par canal, ou comme banque partagée appliquée canal par canal.
    """
    kernels = []
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        kernel = _make_gabor_kernel(ksize, sigma=ksize / 4, theta=theta,
                                     lambd=ksize / 2, gamma=0.5)
        # Normalisation (énergie unitaire) pour une réponse comparable
        # entre orientations
        kernel = kernel / (np.sqrt(np.sum(kernel ** 2)) + 1e-8)
        kernels.append(kernel)

    bank = np.stack(kernels, axis=-1)          # (ksize, ksize, n_orientations)
    bank = bank[:, :, np.newaxis, :]           # (ksize, ksize, 1, n_orientations)
    return bank


class TextureAttentionModule(layers.Layer):
    """
    Module d'attention texturale (TAM) — couche Keras personnalisée,
    insérable après n'importe quel bloc convolutif d'un backbone CNN.

    Args:
        n_orientations : nombre d'orientations de la banque de Gabor
        ksize          : taille des noyaux de Gabor (impair)
        trainable_gate : si True, la couche de gating (conv 1x1 + sigmoid)
                         est entraînable (recommandé) ; les filtres de
                         Gabor eux-mêmes restent toujours fixes.
    """

    def __init__(self, n_orientations: int = 8, ksize: int = 7,
                 trainable_gate: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.n_orientations = n_orientations
        self.ksize = ksize
        self.trainable_gate = trainable_gate

    def build(self, input_shape):
        self.n_channels = input_shape[-1]
        gabor_bank = _build_gabor_bank(self.ksize, self.n_orientations)

        # Un filtre de Gabor partagé, appliqué à CHAQUE canal d'entrée
        # indépendamment (depthwise), produisant n_orientations réponses
        # par canal — on agrège ensuite par max sur les orientations.
        self.gabor_weights = self.add_weight(
            name="gabor_bank",
            shape=(self.ksize, self.ksize, self.n_channels, self.n_orientations),
            initializer=keras.initializers.Constant(
                np.tile(gabor_bank, (1, 1, self.n_channels, 1))
            ),
            trainable=False,   # filtres de Gabor fixes (garantit l'interprétabilité)
        )

        # Gating spatial appris : compresse l'énergie texturale multi-
        # orientations en un masque d'attention (H, W, 1)
        self.gate_conv = layers.Conv2D(
            1, kernel_size=1, activation="sigmoid",
            trainable=self.trainable_gate, name="tam_gate"
        )
        super().build(input_shape)

    def call(self, x):
        # Réponse de chaque orientation, pour chaque canal, via depthwise
        # conv généralisée : on traite chaque (canal x orientation) comme
        # un filtre depthwise séparé, puis on regroupe.
        responses = tf.nn.depthwise_conv2d(
            x, self.gabor_weights, strides=[1, 1, 1, 1], padding="SAME"
        )
        # responses : (B, H, W, n_channels * n_orientations) — TF ordonne
        # les canaux de sortie comme [canal0_orient0..K, canal1_orient0..K, ...],
        # donc un simple reshape vers (B, H, W, n_channels, n_orientations)
        # restaure la structure attendue sans permutation supplémentaire.
        shape = tf.shape(responses)
        responses = tf.reshape(
            responses, [shape[0], shape[1], shape[2], self.n_channels, self.n_orientations]
        )

        # Magnitude + agrégation MAX sur les orientations (réponse
        # invariante à l'orientation dominante de la texture locale)
        energy = tf.reduce_max(tf.abs(responses), axis=-1)   # (B, H, W, C)

        # Compression canal → carte d'énergie texturale globale
        energy_map = tf.reduce_mean(energy, axis=-1, keepdims=True)  # (B, H, W, 1)

        # Masque d'attention spatial appris à partir de l'énergie texturale
        attention_mask = self.gate_conv(energy_map)  # (B, H, W, 1)

        # Gating résiduel : préserve le signal original, amplifie les
        # régions à forte réponse texturale
        return x + x * attention_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_orientations": self.n_orientations,
            "ksize": self.ksize,
            "trainable_gate": self.trainable_gate,
        })
        return config


# ─────────────────────────────────────────────────────────────
# Architectures hybrides — backbone + TAM
# ─────────────────────────────────────────────────────────────

def build_resnet50_tam(input_shape: tuple, num_classes: int,
                        freeze_base: bool = True,
                        tam_stages: tuple = ("conv3_block4_out", "conv4_block6_out"),
                        n_orientations: int = 8) -> keras.Model:
    """
    ResNet50 augmenté de modules TAM insérés après des blocs résiduels
    intermédiaires (par défaut : fin des stages conv3 et conv4 — résolution
    intermédiaire, où les motifs texturaux locaux sont les plus discriminants
    avant l'abstraction sémantique des couches profondes).

    Args:
        input_shape    : (H, W, C)
        num_classes    : nombre de classes de sortie
        freeze_base    : gèle le backbone ResNet50 (transfer learning, stage 1)
        tam_stages     : noms des couches de sortie de ResNet50 après
                         lesquelles insérer un TAM (voir keras.applications
                         .ResNet50(...).summary() pour les noms exacts)
        n_orientations : nombre d'orientations de Gabor dans chaque TAM

    Returns:
        Modèle Keras non compilé, avec model.base_model pour le fine-tuning
    """
    base_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)

    # Construction d'un modèle intermédiaire exposant les sorties des
    # stages ciblés, pour pouvoir insérer TAM après chacune d'elles.
    stage_outputs = {name: base_model.get_layer(name).output for name in tam_stages}
    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=[base_model.output] + list(stage_outputs.values()),
        name="resnet50_multi_stage"
    )

    outputs_list = feature_extractor(x)
    final_features = outputs_list[0]
    intermediate_features = outputs_list[1:]

    # Applique un TAM sur chaque feature intermédiaire, puis les réinjecte
    # via une branche auxiliaire globale (GAP) concaténée aux features
    # finales — permet au TAM d'influencer la décision même si son
    # emplacement est en milieu de réseau.
    tam_branches = []
    for i, feat in enumerate(intermediate_features):
        tam_out = TextureAttentionModule(
            n_orientations=n_orientations, name=f"tam_stage_{i}"
        )(feat)
        pooled = layers.GlobalAveragePooling2D(name=f"tam_gap_{i}")(tam_out)
        tam_branches.append(pooled)

    final_pooled = layers.GlobalAveragePooling2D(name="final_gap")(final_features)

    x = layers.Concatenate(name="tam_fusion")([final_pooled] + tam_branches)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ResNet50_TAM")
    model.base_model = base_model
    return model


def build_efficientnetb0_tam(input_shape: tuple, num_classes: int,
                              freeze_base: bool = True,
                              tam_stages: tuple = ("block4a_expand_activation",
                                                     "block6a_expand_activation"),
                              n_orientations: int = 8) -> keras.Model:
    """
    EfficientNetB0 augmenté de modules TAM — même principe que
    build_resnet50_tam, adapté aux noms de couches d'EfficientNetB0.
    """
    base_model = keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0)(inputs)

    stage_outputs = {name: base_model.get_layer(name).output for name in tam_stages}
    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=[base_model.output] + list(stage_outputs.values()),
        name="efficientnetb0_multi_stage"
    )

    outputs_list = feature_extractor(x)
    final_features = outputs_list[0]
    intermediate_features = outputs_list[1:]

    tam_branches = []
    for i, feat in enumerate(intermediate_features):
        tam_out = TextureAttentionModule(
            n_orientations=n_orientations, name=f"tam_stage_{i}"
        )(feat)
        pooled = layers.GlobalAveragePooling2D(name=f"tam_gap_{i}")(tam_out)
        tam_branches.append(pooled)

    final_pooled = layers.GlobalAveragePooling2D(name="final_gap")(final_features)

    x = layers.Concatenate(name="tam_fusion")([final_pooled] + tam_branches)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_TAM")
    model.base_model = base_model
    return model