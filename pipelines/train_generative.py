"""
pipelines/train_generative.py
--------------------------------
Phase 3 — Étape 2 : Entraînement cGAN + VAE sur les classes minoritaires.

Pour chaque classe jugée minoritaire (ratio < threshold_ratio de la classe
majoritaire) :
  1. Construction d'un petit dataset d'images réelles de cette classe
  2. Entraînement du cGAN (Generator + Discriminator, boucle GAN classique)
  3. Entraînement du VAE conditionnel
  4. Génération d'images synthétiques avec les 2 modèles
  5. Évaluation : FID score, SSIM, grille d'aperçu visuel
  6. Sauvegarde des poids, images générées, métriques

Usage :
    from pipelines.train_generative import run_generative_phase
    results = run_generative_phase(split_result, config)
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from PIL import Image

from pipelines.models_generative import build_cgan, build_cvae


# ─────────────────────────────────────────────────────────────
# Identification des classes minoritaires
# ─────────────────────────────────────────────────────────────

def _identify_minority_classes(train_df: pd.DataFrame, idx2label: dict,
                                threshold_ratio: float) -> list:
    """
    Identifie les classes dont le nombre d'images est sous le seuil
    (threshold_ratio * nombre d'images de la classe majoritaire).

    Returns:
        Liste de dicts { label, label_idx, count } triée par count croissant
    """
    counts = train_df["label_idx"].value_counts()
    max_count = counts.max()
    threshold = threshold_ratio * max_count

    minority = []
    for idx, count in counts.items():
        if count < threshold:
            minority.append({
                "label_idx": int(idx),
                "label"    : idx2label[int(idx)],
                "count"    : int(count)
            })

    minority.sort(key=lambda x: x["count"])
    return minority


# ─────────────────────────────────────────────────────────────
# Chargement d'images pour une classe donnée
# ─────────────────────────────────────────────────────────────

def _load_class_images(df: pd.DataFrame, label_idx: int,
                        img_size: tuple) -> np.ndarray:
    """
    Charge toutes les images d'une classe, redimensionnées à img_size,
    normalisées en [-1, 1] (pour le générateur GAN avec activation tanh).

    Returns:
        np.ndarray de forme (N, H, W, 3)
    """
    class_df = df[df["label_idx"] == label_idx]
    images = []

    for fp in class_df["filepath"]:
        try:
            img = Image.open(fp).convert("RGB").resize((img_size[1], img_size[0]))
            arr = np.array(img, dtype=np.float32)
            images.append(arr)
        except Exception:
            continue

    images = np.stack(images, axis=0)
    return images


# ─────────────────────────────────────────────────────────────
# Entraînement cGAN sur une seule classe
# ─────────────────────────────────────────────────────────────

def _train_cgan_one_class(images: np.ndarray, label_idx: int,
                           num_classes: int, config: dict,
                           output_dir: Path) -> dict:
    """
    Entraîne un cGAN sur les images d'une seule classe minoritaire
    (le conditionnement multi-classe permet de réutiliser la même
    architecture, mais l'entraînement ici se concentre sur une classe
    à la fois pour un meilleur contrôle de la qualité par classe).
    """
    cfg = config["phase3"]["cgan"]
    img_size = tuple(config["preprocessing"]["img_size"][:2]) + (3,)
    # On réduit la résolution pour le GAN (coût computationnel) si configuré
    gan_img_size = (64, 64, 3)

    latent_dim = cfg["latent_dim"]
    batch_size = min(cfg["batch_size"], len(images))

    # Resize vers la résolution GAN + normalisation [-1, 1]
    images_resized = tf.image.resize(images, gan_img_size[:2]).numpy()
    images_norm = (images_resized / 127.5) - 1.0

    generator, discriminator = build_cgan(latent_dim, num_classes, gan_img_size)

    g_optimizer = keras.optimizers.Adam(cfg["g_learning_rate"], beta_1=cfg["beta_1"])
    d_optimizer = keras.optimizers.Adam(cfg["d_learning_rate"], beta_1=cfg["beta_1"])

    bce = keras.losses.BinaryCrossentropy()
    label_smoothing = cfg.get("label_smoothing", 0.0)

    labels_batch = np.full((batch_size, 1), label_idx, dtype=np.int32)

    @tf.function
    def train_step(real_images, real_labels):
        noise = tf.random.normal((batch_size, latent_dim))

        # --- Discriminateur ---
        with tf.GradientTape() as d_tape:
            fake_images = generator([noise, real_labels], training=True)

            real_pred = discriminator([real_images, real_labels], training=True)
            fake_pred = discriminator([fake_images, real_labels], training=True)

            real_target = tf.ones_like(real_pred) * (1.0 - label_smoothing)
            fake_target = tf.zeros_like(fake_pred)

            d_loss_real = bce(real_target, real_pred)
            d_loss_fake = bce(fake_target, fake_pred)
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        # --- Générateur ---
        with tf.GradientTape() as g_tape:
            fake_images = generator([noise, real_labels], training=True)
            fake_pred = discriminator([fake_images, real_labels], training=True)
            g_loss = bce(tf.ones_like(fake_pred), fake_pred)

        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        return d_loss, g_loss

    history = {"d_loss": [], "g_loss": []}
    n_batches_per_epoch = max(1, len(images_norm) // batch_size)

    print(f"    cGAN — {cfg['epochs']} epochs, batch_size={batch_size}, "
          f"{len(images_norm)} images réelles")

    for epoch in range(cfg["epochs"]):
        idx = np.random.randint(0, len(images_norm), batch_size)
        real_batch = tf.convert_to_tensor(images_norm[idx], dtype=tf.float32)
        real_labels = tf.convert_to_tensor(labels_batch, dtype=tf.int32)

        d_loss, g_loss = train_step(real_batch, real_labels)
        history["d_loss"].append(float(d_loss))
        history["g_loss"].append(float(g_loss))

        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0:
            print(f"      Epoch {epoch+1}/{cfg['epochs']} — "
                  f"D_loss: {d_loss:.4f}  G_loss: {g_loss:.4f}")

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history,
        "img_size": gan_img_size,
    }


# ─────────────────────────────────────────────────────────────
# Entraînement VAE sur une seule classe
# ─────────────────────────────────────────────────────────────

def _train_vae_one_class(images: np.ndarray, label_idx: int,
                          num_classes: int, config: dict) -> dict:
    """
    Entraîne un VAE conditionnel sur les images d'une seule classe minoritaire.
    """
    cfg = config["phase3"]["vae"]
    vae_img_size = (64, 64, 3)
    latent_dim = cfg["latent_dim"]
    batch_size = min(cfg["batch_size"], len(images))

    images_resized = tf.image.resize(images, vae_img_size[:2]).numpy()
    images_norm = images_resized / 255.0   # [0, 1] pour sortie sigmoid

    labels_batch = np.full((len(images_norm), 1), label_idx, dtype=np.int32)

    encoder, decoder, vae = build_cvae(latent_dim, num_classes, vae_img_size,
                                        kl_weight=cfg["kl_weight"])
    vae.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"]))

    print(f"    VAE — {cfg['epochs']} epochs, batch_size={batch_size}, "
          f"{len(images_norm)} images réelles")

    # tf.data.Dataset robuste : chaque élément est (images, labels), consommé
    # directement par train_step/test_step qui attendent ce tuple (pas de y séparé).
    dataset = tf.data.Dataset.from_tensor_slices((
        images_norm.astype(np.float32), labels_batch
    ))
    dataset = dataset.shuffle(buffer_size=len(images_norm)).batch(
        batch_size, drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)

    history = vae.fit(
        dataset,
        epochs=cfg["epochs"],
        verbose=0,
        callbacks=[
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: (
                    print(f"      Epoch {epoch+1}/{cfg['epochs']} — "
                          f"loss: {logs['loss']:.4f}  "
                          f"recon: {logs['reconstruction_loss']:.4f}  "
                          f"kl: {logs['kl_loss']:.4f}")
                    if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0 else None
                )
            )
        ]
    )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "vae": vae,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "img_size": vae_img_size,
    }


# ─────────────────────────────────────────────────────────────
# Génération d'échantillons synthétiques
# ─────────────────────────────────────────────────────────────

def _generate_cgan_samples(generator, label_idx: int, n_samples: int,
                            latent_dim: int) -> np.ndarray:
    """Génère n_samples images via le générateur cGAN. Retourne en [0, 255]."""
    noise = tf.random.normal((n_samples, latent_dim))
    labels = tf.fill((n_samples, 1), label_idx)
    generated = generator([noise, labels], training=False).numpy()
    # tanh → [-1, 1] → [0, 255]
    generated = ((generated + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return generated


def _generate_vae_samples(decoder, label_idx: int, n_samples: int,
                           latent_dim: int) -> np.ndarray:
    """Génère n_samples images via le décodeur VAE. Retourne en [0, 255]."""
    z = tf.random.normal((n_samples, latent_dim))
    labels = tf.fill((n_samples, 1), label_idx)
    generated = decoder([z, labels], training=False).numpy()
    # sigmoid → [0, 1] → [0, 255]
    generated = (generated * 255.0).clip(0, 255).astype(np.uint8)
    return generated


def _save_samples(images: np.ndarray, save_dir: Path, prefix: str) -> None:
    """Sauvegarde un batch d'images générées sur disque."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, img_arr in enumerate(images):
        Image.fromarray(img_arr).save(save_dir / f"{prefix}_{i:04d}.png")


def _plot_sample_grid(images: np.ndarray, title: str, save_path: Path,
                       n_show: int = 16) -> None:
    """Génère une grille d'aperçu des images synthétiques."""
    n_show = min(n_show, len(images))
    cols = 4
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i in range(rows * cols):
        ax = axes[i] if rows > 1 or cols > 1 else axes
        if i < n_show:
            ax.imshow(images[i])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Évaluation — FID & SSIM
# ─────────────────────────────────────────────────────────────

def _compute_fid(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """
    Calcule le FID score (Fréchet Inception Distance) entre images réelles
    et générées, en utilisant InceptionV3 comme extracteur de features.

    NB: nécessite un minimum d'images (idéalement ≥ 50) pour une estimation
    stable de la covariance.
    """
    from scipy import linalg

    inception = keras.applications.InceptionV3(
        include_top=False, pooling="avg", input_shape=(299, 299, 3)
    )

    def get_features(images):
        images_resized = tf.image.resize(images, (299, 299))
        images_preprocessed = keras.applications.inception_v3.preprocess_input(
            images_resized.numpy().astype(np.float32)
        )
        return inception.predict(images_preprocessed, verbose=0)

    real_features = get_features(real_images)
    fake_features = get_features(fake_images)

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def _compute_ssim(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """
    Calcule le SSIM moyen entre paires aléatoires d'images réelles/générées
    (mesure de similarité structurelle).
    """
    n = min(len(real_images), len(fake_images))
    real_t = tf.image.resize(real_images[:n], fake_images.shape[1:3])
    real_t = tf.cast(real_t, tf.float32)
    fake_t = tf.cast(fake_images[:n], tf.float32)

    ssim_values = tf.image.ssim(real_t, fake_t, max_val=255.0)
    return float(tf.reduce_mean(ssim_values))


# ─────────────────────────────────────────────────────────────
# Orchestrateur — une classe minoritaire complète
# ─────────────────────────────────────────────────────────────

def _process_one_minority_class(cls_info: dict, train_df: pd.DataFrame,
                                 num_classes: int, config: dict,
                                 output_dir: Path) -> dict:
    """
    Pipeline complet pour UNE classe minoritaire :
    chargement → cGAN → VAE → génération → évaluation → sauvegarde.
    """
    label = cls_info["label"]
    label_idx = cls_info["label_idx"]
    count = cls_info["count"]

    print(f"\n{'='*55}")
    print(f"  🧬 Classe minoritaire : {label} ({count} images réelles)")
    print(f"{'='*55}")

    img_size = tuple(config["preprocessing"]["img_size"])
    class_dir = output_dir / label
    class_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des images réelles de cette classe
    print(f"\n  [1/4] Chargement des images réelles...")
    real_images = _load_class_images(train_df, label_idx, img_size)
    print(f"  ✅ {len(real_images)} images chargées")

    if len(real_images) < 10:
        print(f"  ⚠️  Trop peu d'images ({len(real_images)}) — classe ignorée.")
        return None

    result = {"label": label, "label_idx": label_idx, "real_count": len(real_images)}

    # 2. cGAN
    print(f"\n  [2/4] Entraînement cGAN...")
    t0 = time.time()
    cgan_result = _train_cgan_one_class(real_images, label_idx, num_classes,
                                         config, class_dir)
    cgan_time = time.time() - t0
    print(f"  ⏱️  cGAN entraîné en {cgan_time/60:.1f} min")

    cgan_cfg = config["phase3"]["cgan"]
    cgan_samples = _generate_cgan_samples(
        cgan_result["generator"], label_idx,
        cgan_cfg["n_samples_per_class"], cgan_cfg["latent_dim"]
    )
    _save_samples(cgan_samples, class_dir / "cgan_synthetic", "cgan")
    _plot_sample_grid(cgan_samples, f"cGAN — {label}",
                      class_dir / "cgan_preview.png")
    cgan_result["generator"].save(class_dir / "cgan_generator.keras")

    # 3. VAE
    print(f"\n  [3/4] Entraînement VAE conditionnel...")
    t0 = time.time()
    vae_result = _train_vae_one_class(real_images, label_idx, num_classes, config)
    vae_time = time.time() - t0
    print(f"  ⏱️  VAE entraîné en {vae_time/60:.1f} min")

    vae_cfg = config["phase3"]["vae"]
    vae_samples = _generate_vae_samples(
        vae_result["decoder"], label_idx,
        vae_cfg["n_samples_per_class"], vae_cfg["latent_dim"]
    )
    _save_samples(vae_samples, class_dir / "vae_synthetic", "vae")
    _plot_sample_grid(vae_samples, f"VAE — {label}",
                      class_dir / "vae_preview.png")
    vae_result["decoder"].save(class_dir / "vae_decoder.keras")

    # 4. Évaluation
    print(f"\n  [4/4] Évaluation (FID, SSIM)...")
    eval_cfg = config["phase3"]["evaluation"]
    n_eval = min(eval_cfg["n_eval_samples"], len(real_images))
    real_eval = real_images[:n_eval]

    metrics = {"cgan": {}, "vae": {}}

    if eval_cfg.get("compute_ssim", True):
        metrics["cgan"]["ssim"] = round(_compute_ssim(real_eval, cgan_samples[:n_eval]), 4)
        metrics["vae"]["ssim"]  = round(_compute_ssim(real_eval, vae_samples[:n_eval]), 4)
        print(f"  SSIM  — cGAN: {metrics['cgan']['ssim']}  VAE: {metrics['vae']['ssim']}")

    if eval_cfg.get("compute_fid", True):
        try:
            metrics["cgan"]["fid"] = round(_compute_fid(real_eval, cgan_samples[:n_eval]), 2)
            metrics["vae"]["fid"]  = round(_compute_fid(real_eval, vae_samples[:n_eval]), 2)
            print(f"  FID   — cGAN: {metrics['cgan']['fid']}  VAE: {metrics['vae']['fid']}")
        except Exception as e:
            print(f"  ⚠️  FID non calculé ({e})")

    result.update({
        "cgan_train_time_min": round(cgan_time / 60, 2),
        "vae_train_time_min" : round(vae_time / 60, 2),
        "n_synthetic_cgan"   : len(cgan_samples),
        "n_synthetic_vae"    : len(vae_samples),
        "metrics"            : metrics,
    })

    with open(class_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────
# Orchestrateur principal — Phase 3
# ─────────────────────────────────────────────────────────────

def run_generative_phase(split_result: dict, config: dict,
                          classes: list = None) -> dict:
    """
    Pipeline complet de la Phase 3 : identifie les classes minoritaires,
    entraîne cGAN + VAE pour chacune, génère et évalue les échantillons.

    Args:
        split_result : résultat de split_dataset() (Phase 1) — doit contenir
                       'train_df', 'idx2label'
        config       : configuration globale
        classes      : liste explicite de labels à traiter (override),
                       sinon auto-détection via phase3.minority_selection

    Returns:
        dict avec les résultats par classe + résumé
    """
    print("\n" + "=" * 55)
    print("  PHASE 3 — Génération de Données Synthétiques (cGAN + VAE)")
    print("=" * 55)

    train_df    = split_result["train_df"]
    idx2label   = split_result["idx2label"]
    num_classes = config["classes"]["num_classes"]
    output_dir  = Path(config["phase3"]["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_ratio = config["phase3"]["minority_selection"]["threshold_ratio"]

    if classes:
        label2idx = {v: int(k) for k, v in idx2label.items()}
        minority_classes = [
            {"label": c, "label_idx": label2idx[c],
             "count": int((train_df["label_idx"] == label2idx[c]).sum())}
            for c in classes if c in label2idx
        ]
    else:
        minority_classes = _identify_minority_classes(train_df, idx2label, threshold_ratio)

    if not minority_classes:
        print("\n  ℹ️  Aucune classe minoritaire détectée avec le seuil actuel.")
        return {"processed_classes": [], "output_dir": str(output_dir)}

    print(f"\n  Classes minoritaires détectées ({len(minority_classes)}) "
          f"[seuil = {threshold_ratio*100:.0f}% de la classe majoritaire] :")
    for c in minority_classes:
        print(f"    {c['label']:<25} → {c['count']} images")

    all_results = []
    for cls_info in minority_classes:
        result = _process_one_minority_class(
            cls_info, train_df, num_classes, config, output_dir
        )
        if result:
            all_results.append(result)

    # Résumé global
    print("\n" + "=" * 55)
    print("  ✅ RÉSUMÉ — PHASE 3 GÉNÉRATIVE")
    print("=" * 55)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "label": r["label"],
            "real_count": r["real_count"],
            "n_synthetic_cgan": r["n_synthetic_cgan"],
            "n_synthetic_vae": r["n_synthetic_vae"],
            "cgan_ssim": r["metrics"]["cgan"].get("ssim"),
            "vae_ssim": r["metrics"]["vae"].get("ssim"),
            "cgan_fid": r["metrics"]["cgan"].get("fid"),
            "vae_fid": r["metrics"]["vae"].get("fid"),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "phase3_summary.csv", index=False)

    with open(output_dir / "phase3_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  📁 Résultats sauvegardés dans : {output_dir}")
    print("  ✅ Phase 3 (Génératif) terminée\n")

    return {
        "processed_classes": [r["label"] for r in all_results],
        "all_results": all_results,
        "output_dir": str(output_dir),
    }