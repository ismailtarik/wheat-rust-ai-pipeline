"""
evaluation/aggregate_seeds.py
--------------------------------
Agrégation des runs répétés multi-seeds (reproductibilité statistique).

Objectif scientifique :
    Les écarts de F1 observés entre les mécanismes d'attention à leur
    position optimale sont de l'ordre de ~1 point. Avec une seule
    exécution par configuration, il est IMPOSSIBLE de distinguer un effet
    réel d'une variance d'initialisation. Ce script agrège plusieurs runs
    (graines différentes) pour rapporter mean ± std, seule base
    défendable pour comparer des modèles aussi proches.

Prérequis — lancer le même jeu de modèles avec plusieurs graines :
    python main.py --phase 2 --data_source merged --task classification \\
        --no_skip_existing --seed 42   --models "<liste>"
    python main.py --phase 2 --data_source merged --task classification \\
        --no_skip_existing --seed 123  --models "<liste>"
    python main.py --phase 2 --data_source merged --task classification \\
        --no_skip_existing --seed 2024 --models "<liste>"

    La graine 42 écrit dans  outputs/phase2_classification_merged/
    Les autres graines dans  outputs/phase2_classification_merged_seed<N>/
    (isolation gérée par main.py, voir --seed).

Sorties :
    results/statistics/repeated_runs.csv
        Une ligne par (modèle, graine) — les valeurs BRUTES de chaque run.
    results/statistics/repeated_runs_summary.csv
        Une ligne par modèle — mean ± std sur accuracy / macro-F1 /
        weighted-F1, avec le nombre de runs effectivement trouvés.

ANTI-FABRICATION :
    Ce script ne lit QUE des classification_report.json réellement
    présents sur disque. Une graine manquante pour un modèle n'est
    jamais extrapolée : elle est absente du CSV brut et signalée dans
    la console, et le n_runs du résumé reflète le nombre réel de runs.
    Aucun test de significativité n'est calculé ici : avec 3 graines,
    seuls mean ± std sont rapportés, conformément à la consigne du
    projet ("Do NOT perform statistical significance tests incorrectly").

Usage :
    python evaluation/aggregate_seeds.py
    python evaluation/aggregate_seeds.py --seeds 42 123 2024 --models resnet50 resnet50_tam
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [42, 123, 2024]
DEFAULT_SEED = 42   # doit correspondre à _DEFAULT_SEED dans main.py

DEFAULT_MODELS = [
    "resnet50",
    "resnet50_se",
    "resnet50_cbam",
    "resnet50_triplet",
    "resnet50_tam",
]

BASE_OUTPUT_DIR = "outputs/phase2_classification_merged"


def _seed_output_dir(seed: int, base: str = BASE_OUTPUT_DIR) -> Path:
    """Reproduit exactement la convention de nommage de main.py."""
    if seed == DEFAULT_SEED:
        return Path(base)
    return Path(f"{base}_seed{seed}")


def _read_run(model_name: str, seed: int, base: str) -> dict:
    """
    Lit le classification_report.json d'un (modèle, graine) donné.
    Retourne None si le run n'existe pas sur disque.
    """
    report_path = _seed_output_dir(seed, base) / model_name / "classification_report.json"
    if not report_path.exists():
        return None

    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Lecture impossible pour {model_name} / seed {seed} ({e})")
        return None

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    return {
        "model_name": model_name,
        "seed": seed,
        "accuracy": round(float(report.get("accuracy", 0.0)), 4),
        "macro_precision": round(float(macro.get("precision", 0.0)), 4),
        "macro_recall": round(float(macro.get("recall", 0.0)), 4),
        "macro_f1": round(float(macro.get("f1-score", 0.0)), 4),
        "weighted_f1": round(float(weighted.get("f1-score", 0.0)), 4),
    }


def collect_runs(models: list, seeds: list, base: str) -> pd.DataFrame:
    """Rassemble tous les runs réellement présents sur disque."""
    rows, missing = [], []

    for model_name in models:
        for seed in seeds:
            run = _read_run(model_name, seed, base)
            if run is None:
                missing.append((model_name, seed))
            else:
                rows.append(run)

    if missing:
        print(f"\n  ⚠️  {len(missing)} run(s) manquant(s) — non extrapolé(s) :")
        for model_name, seed in missing:
            print(f"      - {model_name} (seed {seed})")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule mean ± std par modèle. std sur échantillon (ddof=1), NaN si
    un seul run — on ne prétend pas estimer une dispersion avec n=1.
    """
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    rows = []

    for model_name, group in df.groupby("model_name", sort=False):
        row = {"model_name": model_name, "n_runs": len(group)}
        for m in metrics:
            mean = group[m].mean()
            std = group[m].std(ddof=1) if len(group) > 1 else float("nan")
            row[f"{m}_mean"] = round(float(mean), 4)
            row[f"{m}_std"] = round(float(std), 4) if len(group) > 1 else "NA"
            row[f"{m}_formatted"] = (
                f"{mean:.4f} ± {std:.4f}" if len(group) > 1 else f"{mean:.4f} (n=1)"
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Agrège les runs multi-seeds en mean ± std (reproductibilité)."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--base_dir", type=str, default=BASE_OUTPUT_DIR)
    parser.add_argument("--out_dir", type=str, default="results/statistics")
    args = parser.parse_args()

    print("=" * 60)
    print("  AGRÉGATION MULTI-SEEDS")
    print("=" * 60)
    print(f"  Graines  : {args.seeds}")
    print(f"  Modèles  : {args.models}")
    for seed in args.seeds:
        print(f"    seed {seed:>5} -> {_seed_output_dir(seed, args.base_dir)}")

    df = collect_runs(args.models, args.seeds, args.base_dir)

    if df.empty:
        print("\n  ❌ Aucun run trouvé. Lancez d'abord les entraînements "
              "avec --seed, puis relancez ce script.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "repeated_runs.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  ✅ Runs bruts      : {raw_path}  ({len(df)} lignes)")

    summary = summarize(df)
    summary_path = out_dir / "repeated_runs_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  ✅ Résumé mean±std : {summary_path}")

    print("\n" + "=" * 60)
    print("  RÉSUMÉ — mean ± std (écart-type sur échantillon, ddof=1)")
    print("=" * 60)
    display_cols = ["model_name", "n_runs", "accuracy_formatted",
                     "macro_f1_formatted", "weighted_f1_formatted"]
    print("\n" + summary[display_cols].to_string(index=False))

    n_incomplete = int((summary["n_runs"] < len(args.seeds)).sum())
    if n_incomplete:
        print(f"\n  ⚠️  {n_incomplete} modèle(s) avec moins de {len(args.seeds)} runs — "
              f"interpréter leur écart-type avec prudence.")
    if (summary["n_runs"] < 2).any():
        print("  ⚠️  Certains modèles n'ont qu'un seul run : aucun écart-type "
              "calculable (NA), pas de comparaison statistique possible.")

    print("\n  ℹ️  Aucun test de significativité n'est calculé ici : avec un "
          "petit nombre de graines,\n      seuls mean ± std sont rapportés.")


if __name__ == "__main__":
    main()