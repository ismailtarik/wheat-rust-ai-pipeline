"""Outils de revue visuelle et d'analyse des rapports (planches-contact, validation manuelle).

Usage : python review_tools.py --list | --stage 03 | --stage 03,07a | --stage all
"""
import csv
import math
import sys
import textwrap
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour `import common`

from common import (PROJECT_ROOT)




def stage_03_audit_review():
    """Lecture et analyse des rapports de l'audit global corrige.
    
    Origine : cellule(s) 58, 59 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : data/reports/ ; data/reports/
    """

    # ══════════ cellule 58 ══════════


    ROOT = Path(
        "/content/wheat-rust-ai-pipeline/data/reports/"
    "global_audit_corrected"
    )

    def load_csv(filename):
        path = ROOT / filename
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    conflicts = load_csv("label_conflicts.csv")
    leaks = load_csv("same_dataset_split_leaks.csv")
    cross = load_csv("cross_dataset_duplicates.csv")

    # ============================================================
    # 1. REGROUPEMENT PAR HASH
    # ============================================================

    def group_by_hash(records):
        groups = defaultdict(list)
        for record in records:
            groups[record["sha256"]].append(record)
        return groups

    conflict_groups = group_by_hash(conflicts)
    leak_groups = group_by_hash(leaks)
    cross_groups = group_by_hash(cross)

    # ============================================================
    # 2. ANALYSE DES CONFLITS
    # ============================================================

    print("=" * 70)
    print("ANALYSE DES CONFLITS D'ETIQUETTES")
    print("=" * 70)

    conflict_types = Counter()
    conflict_datasets = Counter()

    for digest, group in conflict_groups.items():

        labels = sorted(set(r["label"] for r in group))
        datasets = sorted(set(r["dataset"] for r in group))

        conflict_types[" + ".join(labels)] += 1

        for dataset in datasets:
            conflict_datasets[dataset] += 1

    print("\nConflits par combinaison de classes :")
    for labels, count in conflict_types.most_common(30):
        print(f"{labels}: {count}")

    print("\nConflits impliquant chaque dataset :")
    for dataset, count in conflict_datasets.most_common():
        print(f"{dataset}: {count}")

    print("\nExemples détaillés :")

    for digest, group in list(conflict_groups.items())[:20]:
        print("\nHash:", digest[:16])

        for r in group:
            print(
                f"  Dataset={r['dataset']} | "
            f"Classe={r['label']} | "
            f"Partition={r['partition']}"
            )
            print("  ", r["path"])

    # ============================================================
    # 3. ANALYSE DES FUITES ENTRE PARTITIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("ANALYSE DES CHEVAUCHEMENTS ENTRE PARTITIONS")
    print("=" * 70)

    leak_combinations = Counter()
    leak_by_dataset = Counter()

    for digest, group in leak_groups.items():

        by_dataset = defaultdict(set)

        for r in group:
            if r["partition"] in {"Train", "Validation", "Test"}:
                by_dataset[r["dataset"]].add(r["partition"])

        for dataset, splits in by_dataset.items():

            if len(splits) > 1:
                combination = " + ".join(sorted(splits))
                leak_combinations[combination] += 1
                leak_by_dataset[dataset] += 1

    print("\nCombinaisons de partitions :")
    for combination, count in leak_combinations.most_common():
        print(f"{combination}: {count}")

    print("\nChevauchements par dataset :")
    for dataset, count in leak_by_dataset.most_common():
        print(f"{dataset}: {count}")

    # ============================================================
    # 4. ANALYSE DES DOUBLONS ENTRE DATASETS
    # ============================================================

    print("\n" + "=" * 70)
    print("ANALYSE DES DOUBLONS INTER-DATASETS")
    print("=" * 70)

    dataset_combinations = Counter()

    for digest, group in cross_groups.items():

        datasets = sorted(set(r["dataset"] for r in group))
        combination = " + ".join(datasets)

        dataset_combinations[combination] += 1

    for combination, count in dataset_combinations.most_common():
        print(f"{combination}: {count}")

    # ============================================================
    # 5. EXPORT DES RAPPORTS DETAILLES
    # ============================================================

    def export_group_summary(groups, filename, mode):

        output = ROOT / filename

        with open(output, "w", newline="", encoding="utf-8") as f:

            fields = [
                "sha256",
                "datasets",
                "labels",
                "partitions",
                "paths",
            ]

            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for digest, group in groups.items():

                writer.writerow({
                    "sha256": digest,
                    "datasets": " | ".join(sorted(
                        set(r["dataset"] for r in group)
                    )),
                    "labels": " | ".join(sorted(
                        set(r["label"] for r in group)
                    )),
                    "partitions": " | ".join(sorted(
                        set(
                            r["dataset"] + ":" + r["partition"]
                            for r in group
                        )
                    )),
                    "paths": " || ".join(
                        r["path"] for r in group
                    ),
                })

        print("Rapport créé :", output)


    export_group_summary(
        conflict_groups,
        "conflict_analysis_detailed.csv",
        "conflict"
    )

    export_group_summary(
        leak_groups,
        "leak_analysis_detailed.csv",
        "leak"
    )

    export_group_summary(
        cross_groups,
        "cross_dataset_analysis_detailed.csv",
        "cross"
    )

    print("\nAnalyse terminée.")

    # ══════════ cellule 59 ══════════


    ROOT = Path(
        "/content/wheat-rust-ai-pipeline/data/reports/"
    "global_audit_corrected"
    )

    INPUT = ROOT / "cross_dataset_duplicates.csv"
    OUTPUT = ROOT / "cross_dataset_label_agreement.csv"

    groups = defaultdict(list)

    with open(INPUT, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["sha256"]].append(row)

    agreement = Counter()
    details = []

    for digest, records in groups.items():

        datasets = {r["dataset"] for r in records}

        if not {"data", "ZenodoWheatSmall"}.issubset(datasets):
            continue

        data_labels = sorted({
            r["label"] for r in records if r["dataset"] == "data"
        })

        zenodo_labels = sorted({
            r["label"] for r in records
            if r["dataset"] == "ZenodoWheatSmall"
        })

        same = set(data_labels) == set(zenodo_labels)

        status = "SAME_LABEL" if same else "LABEL_CONFLICT"
        agreement[status] += 1

        details.append({
            "sha256": digest,
            "data_labels": " | ".join(data_labels),
            "zenodo_labels": " | ".join(zenodo_labels),
            "status": status,
            "data_paths": " || ".join(
                r["path"] for r in records if r["dataset"] == "data"
            ),
            "zenodo_paths": " || ".join(
                r["path"] for r in records
                if r["dataset"] == "ZenodoWheatSmall"
            )
        })

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

        fields = [
            "sha256",
            "data_labels",
            "zenodo_labels",
            "status",
            "data_paths",
            "zenodo_paths"
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)

    print("=" * 60)
    print("DATA / ZENODO LABEL AGREEMENT")
    print("=" * 60)

    print("Groupes communs :", len(details))
    print("Même étiquette :", agreement["SAME_LABEL"])
    print("Étiquettes différentes :", agreement["LABEL_CONFLICT"])

    print("\nExemples de conflits :")

    for item in [x for x in details if x["status"] == "LABEL_CONFLICT"][:20]:
        print(
            f"\nData: {item['data_labels']} "
        f"| Zenodo: {item['zenodo_labels']}"
        )
        print("Data:", item["data_paths"])
        print("Zenodo:", item["zenodo_paths"])

    print("\nRapport :", OUTPUT)


def stage_07a_phash_contact_sheets():
    """Planches-contact des paires de quasi-doublons (revue visuelle).
    
    Origine : cellule(s) 77 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/reports/near_duplicate_audit
    """

    # ══════════ cellule 77 ══════════


    # ============================================================
    # CONFIGURATION
    # ============================================================


    REPORT_DIR = (
        PROJECT_ROOT / "data/reports/near_duplicate_audit"
    )

    PAIRS_PATH = REPORT_DIR / "near_duplicate_pairs.csv"

    GALLERY_DIR = REPORT_DIR / "visual_review"
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    # Nombre de paires par planche
    PAIRS_PER_SHEET = 20

    # Dimensions des miniatures
    THUMB_W = 180
    THUMB_H = 135

    # ============================================================
    # 1. CHARGEMENT DES PAIRES
    # ============================================================

    pairs = pd.read_csv(PAIRS_PATH)

    cross = pairs[pairs["cross_split"] == True].copy()

    cross = cross.sort_values(
        ["hamming_distance", "same_class"],
        ascending=[True, True]
    ).reset_index(drop=True)

    exact_cross = cross[
        cross["hamming_distance"] == 0
    ].copy()

    print("=" * 65)
    print("VISUAL REVIEW — CROSS-PARTITION PAIRS")
    print("=" * 65)

    print("Toutes les paires :", len(pairs))
    print("Paires cross-split :", len(cross))
    print("Paires cross-split avec distance 0 :", len(exact_cross))
    print("Paires cross-split avec classes différentes :",
          int((~cross["same_class"]).sum()))

    print("\nRépartition par distance :")
    print(
        cross["hamming_distance"]
        .value_counts()
        .sort_index()
    )

    print("\nRépartition par classe :")
    print(
        cross.groupby(["class_1", "class_2"])
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    # Export des sous-ensembles
    cross.to_csv(
        REPORT_DIR / "cross_split_candidates.csv",
        index=False
    )

    exact_cross.to_csv(
        REPORT_DIR / "cross_split_exact_phash.csv",
        index=False
    )

    # ============================================================
    # 2. FONCTION DE MINIATURE
    # ============================================================

    def make_thumbnail(path):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((THUMB_W, THUMB_H))

            canvas = Image.new(
                "RGB",
                (THUMB_W, THUMB_H),
                "white"
            )

            x = (THUMB_W - img.width) // 2
            y = (THUMB_H - img.height) // 2

            canvas.paste(img, (x, y))
            return canvas

    # ============================================================
    # 3. CRÉATION DES PLANCHES
    # ============================================================

    def create_contact_sheet(data, output_prefix):

        total = len(data)

        if total == 0:
            print(f"Aucune paire à afficher pour {output_prefix}.")
            return

        for start in range(0, total, PAIRS_PER_SHEET):

            subset = data.iloc[
                start:start + PAIRS_PER_SHEET
            ]

            rows = len(subset)

            margin = 15
            label_h = 45
            row_h = THUMB_H + label_h + 12

            sheet_w = margin * 2 + 2 * THUMB_W + 30
            sheet_h = margin * 2 + rows * row_h

            sheet = Image.new(
                "RGB",
                (sheet_w, sheet_h),
                "#eeeeee"
            )

            draw = ImageDraw.Draw(sheet)

            for i, (_, pair) in enumerate(subset.iterrows()):

                y = margin + i * row_h

                img1 = make_thumbnail(pair["image_1"])
                img2 = make_thumbnail(pair["image_2"])

                x1 = margin
                x2 = margin + THUMB_W + 30

                sheet.paste(img1, (x1, y))
                sheet.paste(img2, (x2, y))

                text = (
                    f"Pair {start + i + 1} | "
                f"Distance={pair['hamming_distance']} | "
                f"{pair['class_1']} / {pair['class_2']}"
                )

                split_text = (
                    f"{pair['split_1']} -> {pair['split_2']}"
                )

                draw.text(
                    (margin, y + THUMB_H + 3),
                    text,
                    fill="black"
                )

                draw.text(
                    (margin, y + THUMB_H + 20),
                    split_text,
                    fill="black"
                )

            sheet_path = (
                GALLERY_DIR
                / f"{output_prefix}_{start // PAIRS_PER_SHEET + 1:03d}.jpg"
            )

            sheet.save(sheet_path, quality=90)

            print("Planche créée :", sheet_path)

    # ============================================================
    # 4. PLANCHES DES PHASH EXACTS ENTRE PARTITIONS
    # ============================================================

    create_contact_sheet(
        exact_cross,
        "cross_split_phash_distance_0"
    )

    print("\nNombre de planches générées :",
          len(list(GALLERY_DIR.glob("*.jpg"))))

    print("\nDossier :", GALLERY_DIR)
    print("Vérification visuelle préparée.")


def stage_09b_cross_class_review_sheets():
    """Verification des chemins et generation des planches de revue inter-classes.
    
    Origine : cellule(s) 90 de "wheat_rust_ai_pipeline (1).ipynb".
    Extraction VERBATIM : code non modifie, sauf !cmd -> subprocess.run, !pip et %magic commentes.
    
    ATTENTION : A VALIDER : le choix de cette version/ce role repose sur une deduction (voir CELL_MAP.md).
    
    Chemins references : /content/wheat-rust-ai-pipeline ; data/processed/classification_v2
    """

    # ══════════ cellule 90 ══════════


    # ============================================================
    # 1. CHEMINS
    # ============================================================

    BASE_DIR = Path("/content/wheat-rust-ai-pipeline")
    V2_DIR = BASE_DIR / "data/processed/classification_v2"
    AUDIT_DIR = V2_DIR / "integrity_audit"

    MANIFEST_PATH = V2_DIR / "classification_manifest.csv"
    GROUP_PATH = AUDIT_DIR / "cross_class_connected_groups.csv"

    OUTPUT_DIR = AUDIT_DIR / "connected_groups_visual_review"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    THUMB_W = 320
    THUMB_H = 220
    COLS = 3
    CELL_H = 310
    MARGIN = 20
    GAP = 15

    # ============================================================
    # 2. CHARGEMENT
    # ============================================================

    manifest = pd.read_csv(MANIFEST_PATH)
    groups = pd.read_csv(GROUP_PATH)

    manifest["sha256"] = manifest["sha256"].astype(str).str.strip()
    groups["sha256"] = groups["sha256"].astype(str).str.strip()

    path_col = (
        "destination_path_v2"
        if "destination_path_v2" in manifest.columns
        else "image_path"
    )

    # Source unique des métadonnées : le manifeste
    manifest_info = (
        manifest[
            ["sha256", "canonical_class", "new_split", path_col]
        ]
        .drop_duplicates(subset="sha256")
        .copy()
    )

    # Vérifier qu'une empreinte ne correspond pas à plusieurs lignes
    if manifest_info["sha256"].duplicated().any():
        raise ValueError("Empreintes SHA-256 ambiguës dans le manifeste.")

    # Fusion sans doublonner les colonnes de classes
    inventory = groups[
        ["group_id", "sha256"]
    ].drop_duplicates().merge(
        manifest_info,
        on="sha256",
        how="left",
        validate="many_to_one"
    )

    if inventory["canonical_class"].isna().any():
        raise ValueError("Certaines classes sont absentes du manifeste.")

    if inventory["new_split"].isna().any():
        raise ValueError("Certaines partitions sont absentes du manifeste.")

    # ============================================================
    # 3. VÉRIFICATION DES CHEMINS
    # ============================================================

    inventory["resolved_path"] = inventory[path_col].apply(
        lambda x: str(x) if pd.notna(x) and Path(str(x)).exists() else None
    )

    missing = inventory[inventory["resolved_path"].isna()]

    print("Images référencées :", len(inventory))
    print("Images accessibles :", inventory["resolved_path"].notna().sum())
    print("Images introuvables :", len(missing))

    if len(missing):
        missing.to_csv(
            OUTPUT_DIR / "missing_group_images.csv",
            index=False
        )
        raise RuntimeError("Images manquantes : génération interrompue.")

    # ============================================================
    # 4. GÉNÉRATION DES PLANCHES
    # ============================================================

    def make_thumbnail(path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_W, THUMB_H))
            return im.copy()

    summary_rows = []

    for group_id, group_df in inventory.groupby("group_id"):

        group_df = group_df.reset_index(drop=True)
        n = len(group_df)

        n_rows = math.ceil(n / COLS)

        page_w = MARGIN * 2 + COLS * THUMB_W + (COLS - 1) * GAP
        page_h = 100 + n_rows * (CELL_H + GAP)

        sheet = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(sheet)

        classes = sorted(group_df["canonical_class"].unique())
        splits = sorted(group_df["new_split"].unique())

        draw.text((MARGIN, 20), f"{group_id} | {n} images", fill="black")
        draw.text(
            (MARGIN, 45),
            "Classes : " + ", ".join(classes),
            fill="black"
        )
        draw.text(
            (MARGIN, 70),
            "Partitions : " + ", ".join(splits),
            fill="black"
        )

        for idx, row in group_df.iterrows():

            r = idx // COLS
            c = idx % COLS

            x = MARGIN + c * (THUMB_W + GAP)
            y = 100 + r * (CELL_H + GAP)

            draw.rectangle(
                [x, y, x + THUMB_W, y + CELL_H],
                outline="gray",
                width=1
            )

            try:
                thumb = make_thumbnail(row["resolved_path"])

                px = x + (THUMB_W - thumb.width) // 2
                py = y + 5

                sheet.paste(thumb, (px, py))

                text_y = y + THUMB_H + 12

                metadata = [
                    f"Class: {row['canonical_class']}",
                    f"Split: {row['new_split']}",
                    f"SHA: {row['sha256'][:16]}..."
                ]

                filename = Path(row["resolved_path"]).name

                for line in metadata:
                    draw.text((x + 8, text_y), line, fill="black")
                    text_y += 19

                for line in textwrap.wrap(filename, width=42)[:2]:
                    draw.text((x + 8, text_y), line, fill="gray")
                    text_y += 16

            except Exception as e:
                draw.text((x + 8, y + 20), f"Erreur: {e}", fill="red")

        output_path = OUTPUT_DIR / f"{group_id}_contact_sheet.jpg"
        sheet.save(output_path, quality=95)

        summary_rows.append({
            "group_id": group_id,
            "number_of_images": n,
            "classes": " | ".join(classes),
            "partitions": " | ".join(splits),
            "cross_partition": len(splits) > 1,
            "contact_sheet": str(output_path)
        })

        print("Planche générée :", output_path)

    # ============================================================
    # 5. EXPORTS
    # ============================================================

    summary = pd.DataFrame(summary_rows)

    inventory.to_csv(
        OUTPUT_DIR / "connected_groups_image_inventory.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "connected_groups_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("AUDIT VISUEL DES GROUPES TERMINÉ")
    print("=" * 65)

    print("Nombre de groupes :", len(summary))
    print("Nombre d'images distinctes :", inventory["sha256"].nunique())
    print(
        "Groupes présents dans plusieurs partitions :",
        summary["cross_partition"].sum()
    )

    print("\nRésumé :")
    print(summary.to_string(index=False))

    print("\nDossier :", OUTPUT_DIR)
    print("Aucune image du dataset n'a été modifiée.")


STAGES = {
    "03": stage_03_audit_review,
    "07a": stage_07a_phash_contact_sheets,
    "09b": stage_09b_cross_class_review_sheets,
}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--stage", default="all", help="numero(s) d'etape separes par des virgules, ou 'all'")
    ap.add_argument("--list", action="store_true", help="lister les etapes")
    a = ap.parse_args(argv)
    if a.list:
        for k, f in STAGES.items(): print(f"{k:>5}  {f.__name__}  -  {(f.__doc__ or '').strip().splitlines()[0]}")
        return
    wanted = list(STAGES) if a.stage == "all" else [x.strip() for x in a.stage.split(",")]
    for k in wanted:
        if k not in STAGES: raise SystemExit(f"Etape inconnue : {k} (voir --list)")
    for k in wanted:
        print(f"\n{'=' * 70}\n  ETAPE {k} : {STAGES[k].__name__}\n{'=' * 70}")
        STAGES[k]()


if __name__ == "__main__":
    main()
