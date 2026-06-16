import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path


def _load_and_preprocess(filepath: tf.Tensor, label: tf.Tensor,
                          img_size: tuple, num_classes: int) -> tuple:
    """
    Charge une image depuis son chemin, la redimensionne et la normalise.

    Étapes :
      1. Lecture du fichier brut
      2. Décodage JPEG/PNG → tenseur RGB
      3. Cast float32
      4. Resize bilinéaire → img_size
      5. Normalisation → [0, 1]
      6. Label → one-hot encoding

    Args:
        filepath    : chemin vers l'image (tf.string)
        label       : index de classe (tf.int32)
        img_size    : tuple (H, W) cible
        num_classes : nombre de classes

    Returns:
        (image_tensor, label_one_hot)
    """
    raw   = tf.io.read_file(filepath)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, img_size, method="bilinear")
    image = image / 255.0
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor, config: dict) -> tuple:
    """
    Applique des transformations aléatoires d'augmentation.

    Transformations (toutes configurables via config.yaml) :
      - Flip horizontal
      - Flip vertical
      - Luminosité aléatoire
      - Contraste aléatoire
      - Saturation aléatoire
      - Teinte aléatoire
      - Zoom/crop aléatoire (depuis image agrandie)
      - Bruit gaussien léger (simulation conditions terrain)

    Args:
        image  : tenseur image [H, W, 3] normalisé
        label  : tenseur label one-hot
        config : dict de la section 'augmentation' du config.yaml

    Returns:
        (image_augmentée, label)
    """
    aug_cfg  = config["augmentation"]
    img_size = tuple(config["preprocessing"]["img_size"])

    if aug_cfg.get("flip_horizontal", True):
        image = tf.image.random_flip_left_right(image)

    if aug_cfg.get("flip_vertical", True):
        image = tf.image.random_flip_up_down(image)

    if "brightness_delta" in aug_cfg:
        image = tf.image.random_brightness(image, max_delta=aug_cfg["brightness_delta"])

    if "contrast_range" in aug_cfg:
        lo, hi = aug_cfg["contrast_range"]
        image  = tf.image.random_contrast(image, lower=lo, upper=hi)

    if "saturation_range" in aug_cfg:
        lo, hi = aug_cfg["saturation_range"]
        image  = tf.image.random_saturation(image, lower=lo, upper=hi)

    if "hue_delta" in aug_cfg:
        image = tf.image.random_hue(image, max_delta=aug_cfg["hue_delta"])

    # Zoom crop : agrandit puis recadre à la taille d'origine
    zoom = aug_cfg.get("zoom_factor", 1.1)
    zoomed_h = int(img_size[0] * zoom)
    zoomed_w = int(img_size[1] * zoom)
    image = tf.image.resize(image, [zoomed_h, zoomed_w], method="bilinear")
    image = tf.image.random_crop(image, size=[img_size[0], img_size[1], 3])

    # Bruit gaussien (simulation capteurs/terrain)
    noise_std = config["preprocessing"].get("noise_std", 0.02)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_std)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, label


def build_tf_datasets(split_result: dict, config: dict) -> tuple:
    """
    Construit les 3 pipelines tf.data (train / val / test).

    Args:
        split_result : Résultat de split_dataset() — doit contenir
                       'train_df', 'val_df', 'test_df', 'label2idx', 'idx2label'
        config       : Dictionnaire chargé depuis config.yaml

    Returns:
        (train_ds, val_ds, test_ds) — tf.data.Dataset prêts pour model.fit()
    """
    print("=" * 55)
    print("  ÉTAPE 3 — Preprocessing & Pipelines tf.data")
    print("=" * 55)

    seed        = config["project"]["seed"]
    img_size    = tuple(config["preprocessing"]["img_size"])
    batch_size  = config["pipeline"]["batch_size"]
    num_classes = config["classes"]["num_classes"]
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df   = split_result["train_df"]
    val_df     = split_result["val_df"]
    test_df    = split_result["test_df"]
    label2idx  = split_result["label2idx"]
    idx2label  = split_result["idx2label"]

    print(f"\n  Configuration :")
    print(f"    Image size   : {img_size}")
    print(f"    Batch size   : {batch_size}")
    print(f"    Num classes  : {num_classes}")
    print(f"    Augmentation : {config['augmentation']['enabled']}")

    # --- Fonctions de mapping avec arguments capturés ---
    def preprocess_fn(fp, lbl):
        return _load_and_preprocess(fp, lbl, img_size, num_classes)

    def augment_fn(img, lbl):
        return _augment(img, lbl, config)

    # --- Construction d'un pipeline générique ---
    def _build_pipeline(df, augment_data: bool, shuffle: bool) -> tf.data.Dataset:
        filepaths = df["filepath"].values
        labels    = df["label_idx"].values

        ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(df),
                seed=seed,
                reshuffle_each_iteration=True
            )

        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if augment_data and config["augmentation"]["enabled"]:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # --- 3 pipelines ---
    print("\n[1/3] Création du pipeline d'entraînement (avec augmentation)...")
    train_ds = _build_pipeline(train_df, augment_data=True, shuffle=True)
    print(f"  train_ds : {len(train_df)} images → {len(train_ds)} batches")

    print("[2/3] Création du pipeline de validation...")
    val_ds = _build_pipeline(val_df, augment_data=False, shuffle=False)
    print(f"  val_ds   : {len(val_df)} images → {len(val_ds)} batches")

    print("[3/3] Création du pipeline de test...")
    test_ds = _build_pipeline(test_df, augment_data=False, shuffle=False)
    print(f"  test_ds  : {len(test_df)} images → {len(test_ds)} batches")

    # --- Vérification d'un batch ---
    sample_imgs, sample_lbls = next(iter(train_ds))
    print(f"\n   Forme d'un batch — images : {sample_imgs.shape}  "
          f"labels : {sample_lbls.shape}")
    print(f"  Valeurs pixels — min : {sample_imgs.numpy().min():.3f}  "
          f"max : {sample_imgs.numpy().max():.3f}")

    # --- Visualisation : Original vs Augmenté ---
    print("\n  Génération de la figure Original vs Augmenté...")
    _plot_augmentation_preview(
        train_df, label2idx, idx2label, img_size, num_classes, config,
        save_path=reports_dir / "04_augmentation_preview.png",
        seed=seed, n_show=6
    )
    print(f"  04_augmentation_preview.png")

    print("\n  Preprocessing terminé — prêt pour le split & Phase 2\n")

    return train_ds, val_ds, test_ds


def _plot_augmentation_preview(df, label2idx: dict, idx2label: dict,
                                img_size: tuple, num_classes: int,
                                config: dict, save_path: Path,
                                seed: int = 42, n_show: int = 6) -> None:
    """
    Génère une figure comparant les images originales et augmentées.
    """
    sample = df.sample(min(n_show, len(df)), random_state=seed)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 5))
    fig.suptitle("Preprocessing : Original vs Augmenté", fontsize=12,
                 fontweight="bold")

    CLASS_COLORS = {
        "Yellow_Rust": "#F4C542",
        "Brown_Rust":  "#8B4513",
        "Stem_Rust":   "#C0392B",
        "Healthy":     "#27AE60",
    }

    for i, (_, row) in enumerate(sample.iterrows()):
        fp    = row["filepath"]
        label = row["label"]
        lbl   = row["label_idx"]

        # Original preprocessé
        img_orig, lbl_oh = _load_and_preprocess(fp, lbl, img_size, num_classes)
        axes[0][i].imshow(img_orig.numpy())
        axes[0][i].set_title(label if i == 0 else "", fontsize=8,
                              color=CLASS_COLORS.get(label, "#2C3E50"),
                              fontweight="bold")
        axes[0][i].axis("off")
        if i == 0:
            axes[0][i].set_ylabel("Original", fontsize=9, fontweight="bold")

        # Augmenté
        img_aug, _ = _augment(img_orig, lbl_oh, config)
        axes[1][i].imshow(img_aug.numpy())
        axes[1][i].axis("off")
        if i == 0:
            axes[1][i].set_ylabel("Augmenté", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()