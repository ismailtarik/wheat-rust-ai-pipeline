"""
pipelines/models_generative.py
---------------------------------
Phase 3 — Étape 1 : Architectures génératives (cGAN + VAE conditionnel).

Objectif (Obj 2 de la thèse) : générer des images synthétiques réalistes
de maladies du blé pour enrichir les classes minoritaires
(ex: Stem_fly, Black_Rust, Common_Root_Rot, Fusarium_Head_Blight...).

Architectures :
  - cGAN : Générateur (bruit + label → image) + Discriminateur
           (image + label → réel/faux), conditionnés par classe.
  - VAE  : Encodeur (image + label → distribution latente) + Décodeur
           (z + label → image reconstruite), conditionné par classe.

Usage :
    from pipelines.models_generative import build_cgan, build_cvae
    generator, discriminator = build_cgan(latent_dim=128, num_classes=15, img_size=(64,64,3))
    encoder, decoder, vae = build_cvae(latent_dim=128, num_classes=15, img_size=(64,64,3))
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────
# cGAN — Conditional GAN
# ─────────────────────────────────────────────────────────────

def build_cgan_generator(latent_dim: int, num_classes: int,
                          img_size: tuple) -> keras.Model:
    """
    Générateur cGAN : (bruit latent, label de classe) → image synthétique.

    Le label est encodé en embedding puis concaténé au vecteur de bruit
    avant d'être projeté et upsamplé jusqu'à la taille d'image cible.
    """
    h, w, c = img_size
    init_size = h // 8   # 3 upsamplings x2 → /8

    noise_input = keras.Input(shape=(latent_dim,), name="noise")
    label_input = keras.Input(shape=(1,), dtype="int32", name="label")

    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    x = layers.Multiply()([noise_input, label_embedding])

    x = layers.Dense(init_size * init_size * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((init_size, init_size, 256))(x)

    # Upsampling x2 → x2 → x2 (init_size → h)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    outputs = layers.Conv2D(c, 3, padding="same", activation="tanh",
                             name="generated_image")(x)

    model = keras.Model([noise_input, label_input], outputs, name="cGAN_Generator")
    return model


def build_cgan_discriminator(num_classes: int, img_size: tuple) -> keras.Model:
    """
    Discriminateur cGAN : (image, label de classe) → probabilité réel/faux.

    Le label est projeté en carte spatiale et concaténé à l'image en entrée
    (technique standard pour conditionner un discriminateur convolutif).
    """
    h, w, c = img_size

    image_input = keras.Input(shape=img_size, name="image")
    label_input = keras.Input(shape=(1,), dtype="int32", name="label")

    label_embedding = layers.Embedding(num_classes, h * w)(label_input)
    label_embedding = layers.Reshape((h, w, 1))(label_embedding)

    x = layers.Concatenate(axis=-1)([image_input, label_embedding])

    x = layers.Conv2D(32, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid", name="real_or_fake")(x)

    model = keras.Model([image_input, label_input], outputs, name="cGAN_Discriminator")
    return model


def build_cgan(latent_dim: int, num_classes: int, img_size: tuple) -> tuple:
    """
    Factory — construit la paire (générateur, discriminateur) d'un cGAN.

    Returns:
        (generator, discriminator) — modèles Keras non compilés
    """
    generator     = build_cgan_generator(latent_dim, num_classes, img_size)
    discriminator = build_cgan_discriminator(num_classes, img_size)
    return generator, discriminator


# ─────────────────────────────────────────────────────────────
# VAE conditionnel
# ─────────────────────────────────────────────────────────────

class Sampling(layers.Layer):
    """Couche de rééchantillonnage (reparameterization trick) : z = mu + sigma * epsilon."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_cvae_encoder(latent_dim: int, num_classes: int,
                        img_size: tuple) -> keras.Model:
    """
    Encodeur du VAE conditionnel : (image, label) → (z_mean, z_log_var, z).
    """
    h, w, c = img_size

    image_input = keras.Input(shape=img_size, name="image")
    label_input = keras.Input(shape=(1,), dtype="int32", name="label")

    label_embedding = layers.Embedding(num_classes, h * w)(label_input)
    label_embedding = layers.Reshape((h, w, 1))(label_embedding)
    x = layers.Concatenate(axis=-1)([image_input, label_embedding])

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling(name="z")([z_mean, z_log_var])

    encoder = keras.Model([image_input, label_input], [z_mean, z_log_var, z],
                          name="cVAE_Encoder")
    return encoder


def build_cvae_decoder(latent_dim: int, num_classes: int,
                        img_size: tuple) -> keras.Model:
    """
    Décodeur du VAE conditionnel : (z, label) → image reconstruite.
    """
    h, w, c = img_size
    init_size = h // 8

    z_input     = keras.Input(shape=(latent_dim,), name="z")
    label_input = keras.Input(shape=(1,), dtype="int32", name="label")

    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    x = layers.Concatenate()([z_input, label_embedding])

    x = layers.Dense(init_size * init_size * 128, activation="relu")(x)
    x = layers.Reshape((init_size, init_size, 128))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(c, 3, padding="same", activation="sigmoid",
                             name="reconstructed_image")(x)

    decoder = keras.Model([z_input, label_input], outputs, name="cVAE_Decoder")
    return decoder


class ConditionalVAE(keras.Model):
    """
    Modèle VAE conditionnel complet, encapsulant l'encodeur et le décodeur
    avec une boucle d'entraînement personnalisée (train_step) intégrant
    la loss de reconstruction + la divergence KL.
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model,
                 kl_weight: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def call(self, inputs):
        images, labels = inputs
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.decoder([z, labels])
        return reconstruction

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.decoder([z, labels])

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(images, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.decoder([z, labels])

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(images, reconstruction),
                axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def build_cvae(latent_dim: int, num_classes: int, img_size: tuple,
                kl_weight: float = 0.001) -> tuple:
    """
    Factory — construit le VAE conditionnel complet.

    Returns:
        (encoder, decoder, vae_model)
    """
    encoder = build_cvae_encoder(latent_dim, num_classes, img_size)
    decoder = build_cvae_decoder(latent_dim, num_classes, img_size)
    vae = ConditionalVAE(encoder, decoder, kl_weight=kl_weight)
    return encoder, decoder, vae