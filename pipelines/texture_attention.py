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


def _get_register_serializable():
    """
    Localise register_keras_serializable de façon robuste selon la version
    TF/Keras installée — certains environnements exposent keras.saving,
    d'autres non (ex: 'keras._tf_keras.keras' n'a pas toujours .saving).
    Retourne un décorateur no-op en dernier recours plutôt que de faire
    planter tout le module (l'enregistrement améliore la ré-utilisabilité
    du modèle sauvegardé mais n'est pas requis pour l'entraînement).
    """
    try:
        from tensorflow.keras.saving import register_keras_serializable
        return register_keras_serializable
    except ImportError:
        pass
    try:
        from tensorflow.keras.utils import register_keras_serializable
        return register_keras_serializable
    except ImportError:
        pass
    try:
        import keras as _keras_standalone
        return _keras_standalone.saving.register_keras_serializable
    except Exception:
        pass

    def _noop_register(*args, **kwargs):
        def _wrap(cls):
            return cls
        return _wrap
    return _noop_register


register_keras_serializable = _get_register_serializable()


@register_keras_serializable(package="WheatAI_TAM")
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
        bypass         : DIAGNOSTIC uniquement — si True, la couche retourne
                         x inchangé (identité pure, aucun calcul de texture).
                         Sert à isoler si une dégradation de performance vient
                         de la simple insertion d'une couche supplémentaire
                         dans le graphe, ou du calcul de texture lui-même.
    """

    def __init__(self, n_orientations: int = 8, ksize: int = 7,
                 trainable_gate: bool = True, bypass: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.n_orientations = n_orientations
        self.ksize = ksize
        self.trainable_gate = trainable_gate
        self.bypass = bypass

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
        if self.bypass:
            # Mode diagnostic : identité pure, aucun calcul de texture.
            return x

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
            "bypass": self.bypass,
        })
        return config


# ─────────────────────────────────────────────────────────────
# Architectures hybrides — backbone + TAM
# ─────────────────────────────────────────────────────────────

def build_resnet50_tam(input_shape: tuple, num_classes: int,
                        freeze_base: bool = True,
                        n_orientations: int = 8,
                        bypass_tam: bool = False) -> keras.Model:
    """
    ResNet50 augmenté d'un module TAM inséré à UN SEUL point du flux
    principal : juste avant le pooling global, sur la feature map finale
    (déjà sémantiquement riche, issue de toute la hiérarchie du backbone).

    Args (ajout) :
        bypass_tam : DIAGNOSTIC — si True, TAM est inséré dans le graphe
                     mais agit comme une identité pure (aucun calcul de
                     texture). Permet de vérifier si une dégradation de
                     performance vient de la structure du graphe elle-même
                     ou du calcul de texture.

    Design (v2 — corrige une v1 défaillante) : la première version insérait
    TAM sur des features INTERMÉDIAIRES en branches latérales poolées puis
    concaténées à la tête de classification. Cela diluait le signal des
    features finales (déjà fortement discriminantes) avec des features
    intermédiaires bruitées n'ayant jamais traversé les couches profondes
    d'abstraction — résultat : F1 -8 à -11 points vs baseline en ablation.
    Cette v2 modifie directement la feature map finale (insertion unique,
    dans le flux principal, comme SE-block/CBAM), préservant la voie de
    classification originale qui fonctionnait déjà bien.

    Args:
        input_shape    : (H, W, C)
        num_classes    : nombre de classes de sortie
        freeze_base    : gèle le backbone ResNet50 (transfer learning, stage 1)
        n_orientations : nombre d'orientations de Gabor dans le TAM

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
    features = base_model(x, training=False)   # (H/32, W/32, 2048)

    # Insertion unique de TAM sur la feature map finale, dans le flux
    # principal (pas de branche latérale, pas de concaténation).
    features = TextureAttentionModule(
        n_orientations=n_orientations, bypass=bypass_tam, name="tam"
    )(features)

    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ResNet50_TAM")
    model.base_model = base_model
    return model


def build_efficientnetb0_tam(input_shape: tuple, num_classes: int,
                              freeze_base: bool = True,
                              n_orientations: int = 8,
                              bypass_tam: bool = False) -> keras.Model:
    """
    EfficientNetB0 + TAM — même principe corrigé que build_resnet50_tam :
    insertion unique sur la feature map finale, dans le flux principal.
    bypass_tam : voir build_resnet50_tam (mode diagnostic identité pure).
    """
    base_model = keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    features = base_model(x, training=False)

    features = TextureAttentionModule(
        n_orientations=n_orientations, bypass=bypass_tam, name="tam"
    )(features)

    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_TAM")
    model.base_model = base_model
    return model