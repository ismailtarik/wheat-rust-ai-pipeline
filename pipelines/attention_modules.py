"""
pipelines/attention_modules.py
---------------------------------
Mécanismes d'attention EXISTANTS (SE, CBAM, Triplet Attention) — utilisés
comme contrôles expérimentaux standards, PAS comme contribution originale.

Objectif scientifique (étude comparative contrôlée) :
    Le pipeline dispose déjà d'un module d'attention original, le Texture
    Attention Module (TAM, voir pipelines/texture_attention.py), présenté
    comme contribution de la thèse. Avant d'attribuer un gain de
    performance à TAM, il est nécessaire de vérifier si des mécanismes
    d'attention GÉNÉRIQUES et déjà publiés (SE, CBAM, Triplet Attention)
    apportent un gain comparable, moindre, ou nul sur le même benchmark
    hétérogène multi-source. Ce module implémente donc fidèlement ces
    trois mécanismes tels que publiés dans la littérature — aucune
    modification, aucune revendication de nouveauté.

Références :
    SE      : Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    CBAM    : Woo et al., "CBAM: Convolutional Block Attention Module",
              ECCV 2018.
    Triplet : Misra et al., "Rotate to Attend: Convolutional Triplet
              Attention Module", WACV 2021.

Convention d'intégration (identique à TAM, pour une comparaison contrôlée) :
    Chaque mécanisme est inséré à UN SEUL point du flux principal — sur la
    feature map finale du backbone ResNet50 (B, H/32, W/32, 2048), juste
    avant le pooling global — exactement comme TextureAttentionModule.
    C'est la SEULE différence architecturale entre les variantes E0 (aucune
    attention), E1 (SE), E2 (CBAM), E3 (Triplet) : même backbone, même
    tête de classification, même position d'insertion.

Usage :
    from pipelines.attention_modules import (
        SEBlock, CBAMBlock, TripletAttention,
        build_resnet50_se, build_resnet50_cbam, build_resnet50_triplet,
    )
    model = build_resnet50_se(input_shape=(256, 256, 3), num_classes=17)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Réutilise le décorateur de sérialisation déjà résolu de façon robuste
# dans texture_attention.py (gère les variations TF/Keras selon
# l'environnement) — pas de duplication de logique, pas de modification
# du fichier existant.
from pipelines.texture_attention import register_keras_serializable


# ─────────────────────────────────────────────────────────────
# SE — Squeeze-and-Excitation (Hu et al., CVPR 2018)
# ─────────────────────────────────────────────────────────────

@register_keras_serializable(package="WheatAI_Attention")
class SEBlock(layers.Layer):
    """
    Bloc Squeeze-and-Excitation — recalibration purement CANAL par CANAL
    des feature maps.

    Étapes (fidèles à la publication originale) :
      1. Squeeze  : Global Average Pooling spatial → descripteur (B, C)
                    résumant chaque canal en un scalaire.
      2. Excitation : deux couches denses (goulot d'étranglement, ratio
                    de réduction configurable) — FC(C→C/r, ReLU) puis
                    FC(C/r→C, sigmoid) — produisent un poids par canal
                    dans [0, 1].
      3. Recalibration : chaque canal de la feature map d'entrée est
                    multiplié par son poids appris (broadcast spatial).

    Args:
        reduction_ratio : ratio de réduction du goulot d'étranglement
                           (r=16 dans la publication originale, valeur
                           standard reprise ici par défaut).
    """

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(1, channels // self.reduction_ratio)

        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(reduced, activation="relu", use_bias=True,
                                 name="se_fc_reduce")
        self.fc2 = layers.Dense(channels, activation="sigmoid", use_bias=True,
                                 name="se_fc_expand")
        super().build(input_shape)

    def call(self, x):
        # Squeeze : (B, H, W, C) -> (B, 1, 1, C)
        s = self.gap(x)
        # Excitation : (B, 1, 1, C) -> (B, 1, 1, C/r) -> (B, 1, 1, C)
        s = self.fc1(s)
        s = self.fc2(s)
        # Recalibration canal par canal (broadcast spatial automatique)
        return x * s

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


# ─────────────────────────────────────────────────────────────
# CBAM — Convolutional Block Attention Module (Woo et al., ECCV 2018)
# ─────────────────────────────────────────────────────────────

@register_keras_serializable(package="WheatAI_Attention")
class CBAMBlock(layers.Layer):
    """
    CBAM — attention CANAL suivie d'attention SPATIALE, appliquées de
    façon SÉQUENTIELLE (ordre canal → spatial, comme dans la publication
    originale, qui montre expérimentalement que cet ordre est légèrement
    supérieur à spatial → canal).

    Sous-module Channel Attention :
      - Deux descripteurs globaux par canal : Global Average Pooling ET
        Global Max Pooling (contrairement à SE qui n'utilise QUE l'average
        pooling — c'est la différence clé revendiquée par CBAM).
      - Un MLP PARTAGÉ (mêmes poids) à goulot d'étranglement appliqué aux
        deux descripteurs.
      - Somme des deux sorties, puis sigmoid → poids canal (B, 1, 1, C).

    Sous-module Spatial Attention :
      - Compression du canal par average pooling ET max pooling le long
        de l'axe canal → deux cartes (B, H, W, 1), concaténées.
      - Convolution 7×7 (padding 'same') + sigmoid → masque spatial
        (B, H, W, 1).

    Args:
        reduction_ratio : ratio de réduction du MLP partagé de l'attention
                           canal (r=16 par défaut, comme dans la
                           publication originale).
        spatial_kernel_size : taille du noyau de la convolution spatiale
                           (7 dans la publication originale).
    """

    def __init__(self, reduction_ratio: int = 16, spatial_kernel_size: int = 7,
                 **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(1, channels // self.reduction_ratio)

        # --- Channel attention : MLP partagé entre les branches avg/max ---
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.gmp = layers.GlobalMaxPooling2D(keepdims=True)
        self.channel_fc1 = layers.Dense(reduced, activation="relu",
                                         use_bias=True, name="cbam_channel_fc_reduce")
        self.channel_fc2 = layers.Dense(channels, activation=None,
                                         use_bias=True, name="cbam_channel_fc_expand")

        # --- Spatial attention : conv 7x7 sur [avg_pool_c ; max_pool_c] ---
        self.spatial_conv = layers.Conv2D(
            filters=1, kernel_size=self.spatial_kernel_size, padding="same",
            activation="sigmoid", use_bias=False, name="cbam_spatial_conv"
        )
        super().build(input_shape)

    def call(self, x):
        # ── Channel attention (MLP partagé, poids identiques sur les 2 branches) ──
        avg_desc = self.channel_fc2(self.channel_fc1(self.gap(x)))   # (B,1,1,C)
        max_desc = self.channel_fc2(self.channel_fc1(self.gmp(x)))   # (B,1,1,C)
        channel_attn = tf.sigmoid(avg_desc + max_desc)                # (B,1,1,C)
        x = x * channel_attn

        # ── Spatial attention (sur la sortie déjà recalibrée en canal) ──
        avg_map = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B,H,W,1)
        max_map = tf.reduce_max(x, axis=-1, keepdims=True)   # (B,H,W,1)
        spatial_desc = tf.concat([avg_map, max_map], axis=-1)  # (B,H,W,2)
        spatial_attn = self.spatial_conv(spatial_desc)          # (B,H,W,1)
        x = x * spatial_attn

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "spatial_kernel_size": self.spatial_kernel_size,
        })
        return config


# ─────────────────────────────────────────────────────────────
# Triplet Attention (Misra et al., WACV 2021)
# ─────────────────────────────────────────────────────────────

@register_keras_serializable(package="WheatAI_Attention")
class TripletAttention(layers.Layer):
    """
    Triplet Attention — capture les interactions croisées entre les TROIS
    paires de dimensions (Canal-Hauteur, Canal-Largeur, Hauteur-Largeur)
    via trois branches parallèles ("rotation" d'axes), sans réduction de
    dimensionnalité (contrairement à SE/CBAM qui compressent les canaux
    dans un goulot d'étranglement).

    Chaque branche applique le même schéma "AttentionGate" :
      1. Z-Pool : compression d'un axe donné en 2 cartes (max + moyenne
         le long de cet axe) — c'est l'opération "Z-pool" de la
         publication originale.
      2. Convolution 7×7 sur ces 2 cartes → 1 carte, puis sigmoid.
      3. Multiplication de la feature map (dans le plan orthogonal à
         l'axe compressé) par ce masque.

    Les trois branches (implémentées ici en convention "channels-last",
    équivalent fonctionnel exact du "channel-first" de la publication
    originale, juste avec les permutations d'axes adaptées) :
      - Branche H-W (spatiale standard) : Z-pool le long de l'axe CANAL →
        masque (B,H,W,1) → même opération que l'attention spatiale de CBAM.
      - Branche C-W : Z-pool le long de l'axe HAUTEUR → capture
        l'interaction Canal×Largeur.
      - Branche C-H : Z-pool le long de l'axe LARGEUR → capture
        l'interaction Canal×Hauteur.

    Sortie : moyenne des trois branches (comme dans la publication
    originale, mode "no_spatial=False").

    Args:
        kernel_size : taille du noyau des 3 convolutions (7 dans la
                      publication originale).
    """

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Une AttentionGate (conv 7x7 + sigmoid) indépendante par branche,
        # comme dans l'implémentation officielle des auteurs (pas de
        # poids partagés entre branches).
        self.conv_hw = layers.Conv2D(
            1, self.kernel_size, padding="same", activation="sigmoid",
            use_bias=False, name="triplet_gate_hw"
        )
        self.conv_cw = layers.Conv2D(
            1, self.kernel_size, padding="same", activation="sigmoid",
            use_bias=False, name="triplet_gate_cw"
        )
        self.conv_ch = layers.Conv2D(
            1, self.kernel_size, padding="same", activation="sigmoid",
            use_bias=False, name="triplet_gate_ch"
        )
        super().build(input_shape)

    @staticmethod
    def _z_pool(x, axis):
        """Concatène max et moyenne le long de `axis` (l'opération Z-Pool)."""
        max_ = tf.reduce_max(x, axis=axis, keepdims=True)
        avg_ = tf.reduce_mean(x, axis=axis, keepdims=True)
        return tf.concat([max_, avg_], axis=axis)

    def call(self, x):
        # x : (B, H, W, C)

        # ── Branche H-W : Z-pool le long de l'axe canal (attention spatiale
        #    standard, identique en principe à la branche spatiale de CBAM) ──
        z_hw = self._z_pool(x, axis=-1)              # (B,H,W,2)
        attn_hw = self.conv_hw(z_hw)                  # (B,H,W,1)
        out_hw = x * attn_hw

        # ── Branche C-W : Z-pool le long de l'axe hauteur (axis=1) ──
        z_cw = self._z_pool(x, axis=1)                # (B,2,W,C)
        # Le plan de convolution doit être (W,C) avec 2 "canaux" (issus du
        # Z-pool) : on permute (B,2,W,C) -> (B,W,C,2)
        z_cw_t = tf.transpose(z_cw, [0, 2, 3, 1])     # (B,W,C,2)
        attn_cw_t = self.conv_cw(z_cw_t)              # (B,W,C,1)
        # Retour à une forme broadcastable sur x le long de H : (B,1,W,C)
        attn_cw = tf.transpose(attn_cw_t, [0, 3, 1, 2])  # (B,1,W,C)
        out_cw = x * attn_cw

        # ── Branche C-H : Z-pool le long de l'axe largeur (axis=2) ──
        z_ch = self._z_pool(x, axis=2)                # (B,H,2,C)
        z_ch_t = tf.transpose(z_ch, [0, 1, 3, 2])     # (B,H,C,2)
        attn_ch_t = self.conv_ch(z_ch_t)              # (B,H,C,1)
        attn_ch = tf.transpose(attn_ch_t, [0, 1, 3, 2])  # (B,H,1,C)
        out_ch = x * attn_ch

        # Moyenne des trois branches (mode "no_spatial=False" des auteurs)
        return (out_hw + out_cw + out_ch) / 3.0

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


# ─────────────────────────────────────────────────────────────
# Architectures hybrides — ResNet50 + attention
# (même schéma d'insertion que build_resnet50_tam dans texture_attention.py :
#  insertion UNIQUE sur la feature map finale, dans le flux principal)
# ─────────────────────────────────────────────────────────────

def _build_resnet50_with_attention(input_shape: tuple, num_classes: int,
                                    attention_layer: layers.Layer,
                                    model_name: str,
                                    freeze_base: bool = True) -> keras.Model:
    """
    Fabrique interne partagée : ResNet50 pré-entraîné ImageNet + une
    couche d'attention insérée sur la feature map finale (avant GAP),
    puis la même tête de classification que build_resnet50 (baseline)
    dans pipelines/models.py — pour garantir que la SEULE différence
    architecturale entre E0/E1/E2/E3 soit le mécanisme d'attention.
    """
    base_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    features = base_model(x, training=False)   # (H/32, W/32, 2048)

    features = attention_layer(features)

    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=model_name)
    model.base_model = base_model   # référence pour le fine-tuning ultérieur
    return model


def build_resnet50_se(input_shape: tuple, num_classes: int,
                       freeze_base: bool = True,
                       reduction_ratio: int = 16) -> keras.Model:
    """ResNet50 + Squeeze-and-Excitation (voir SEBlock)."""
    return _build_resnet50_with_attention(
        input_shape, num_classes,
        attention_layer=SEBlock(reduction_ratio=reduction_ratio, name="se_block"),
        model_name="ResNet50_SE",
        freeze_base=freeze_base,
    )


def build_resnet50_cbam(input_shape: tuple, num_classes: int,
                         freeze_base: bool = True,
                         reduction_ratio: int = 16,
                         spatial_kernel_size: int = 7) -> keras.Model:
    """ResNet50 + CBAM (voir CBAMBlock)."""
    return _build_resnet50_with_attention(
        input_shape, num_classes,
        attention_layer=CBAMBlock(reduction_ratio=reduction_ratio,
                                   spatial_kernel_size=spatial_kernel_size,
                                   name="cbam_block"),
        model_name="ResNet50_CBAM",
        freeze_base=freeze_base,
    )


def build_resnet50_triplet(input_shape: tuple, num_classes: int,
                            freeze_base: bool = True,
                            kernel_size: int = 7) -> keras.Model:
    """ResNet50 + Triplet Attention (voir TripletAttention)."""
    return _build_resnet50_with_attention(
        input_shape, num_classes,
        attention_layer=TripletAttention(kernel_size=kernel_size, name="triplet_attention"),
        model_name="ResNet50_Triplet",
        freeze_base=freeze_base,
    )