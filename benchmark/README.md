# benchmark/ : version consolidée du benchmark de classification

56 fichiers extraits du notebook → **5 fichiers Python**. Le code de chaque étape est inchangé ;
il est rangé dans une fonction, avec une commande pour lancer une étape ou toutes.

```
benchmark/
├── common.py          constantes partagées entre modules (PROJECT, PROJECT_ROOT)
├── build_dataset.py   construction du benchmark V3 (16 étapes) : audit → dédup → pHash → v1 → v2 → v3
├── review_tools.py    3 outils facultatifs : lecture des rapports, planches-contact (07a, 09b, 03)
├── train_eval.py      ResNet50 baseline (PyTorch) : entraînement, évaluation, erreurs, par source
├── make_figures.py    EDA et figures de l'article
└── requirements.txt   (session neuve : pip install -r benchmark/requirements.txt)
```

## Utilisation

```bash
python build_dataset.py --list            # voir les étapes
python build_dataset.py --stage 04        # une étape
python build_dataset.py --stage 04,05     # plusieurs
```

**Ordre complet de `build_dataset.py`** (chaque étape lit la sortie de la précédente) :

`00` acquisition → `01` Mendeley → `02` audit global → `04` déduplication v1 → `05` construction v1 →
`06` audit pHash → `07b` paires inter-classes · `07c` classification des paires → `07d` registre de revue →
`08` construction v2 → `09a` audit d'intégrité → `09c` validation visuelle → `09d` groupes → `09e` décisions →
`10` construction v3 → `11` vérification v3.

Sur Colab : lancer `00,01,02` avec `%run` (montage de Drive, noyau du notebook), le reste avec `!python`.
Ne pas mettre de commentaire sur la ligne d'un `%run`. Ne lancez pas `--stage all` d'emblée.

## Validé sur vos données réelles (étapes 00 à 05)

| Étape | Attendu (notebook d'origine) | Obtenu avec ce paquet |
|---|---|---|
| 02 | 16 628 images · 12 857 SHA-256 uniques · 256 groupes en conflit | identique |
| 04 | 12 601 conservées · 3 445 doublons · 582 en quarantaine (data 10 887 / Mendeley 1 475 / Zenodo 239) | identique |
| 05 | v1 : 8 820 / 1 890 / 1 891 · 0 erreur · 0 fuite SHA entre partitions | identique |

Étapes 06 à 11 : **non encore testées** sur données réelles. Valeurs attendues : v2 = 12 518 (étape 08),
v3 = 12 484 soit 8 761 / 1 859 / 1 864 (étapes 10-11). Le code contient 62 `assert` qui servent de tests.

## Points de fonctionnement à connaître

- **Étapes qui refusent d'écraser** : 01 (`MendeleyWheat_Cleaned`) et 08 (`classification_v2`) s'arrêtent si leur
  dossier de sortie existe déjà. Pour relancer, supprimez-le d'abord.
- **Étape 08 = nettoyage automatique.** Elle retire les images des paires pHash de distance 0 entre classes
  différentes. Le registre `manual_review_register.csv` doit exister (produit par 07d) mais n'influence pas la décision.
  La revue **manuelle** intervient à v2 → v3 : les observations des 21 paires sont écrites dans le code de `09c`.
- **`train_eval.py`** : l'étape 14 a besoin de `NUM_CLASSES` (étape 13), la 16 de `test_loader` (étape 14).
  Lancez `--stage 13,14,15,16` dans une seule commande. L'étape 13 réentraîne le modèle.
- Les étapes de `build_dataset.py` communiquent par fichiers CSV : elles sont indépendantes en mémoire.

## Ce qui a été vérifié

- Code interne de chaque étape identique à l'original (comparaison d'arbres syntaxiques) ; chaînes multi-lignes intactes.
- Aucun nom indéfini, aucune variable utilisée avant affectation ; import des 4 modules sans effet de bord.
- Consolidation validée sur un pipeline d'essai (sorties et fichiers identiques avant/après).

## À savoir sur la taille

Environ 7 700 lignes, comme avant : les noms répétés dans le notebook (`OUTPUT_DIR` ×11, `BASE_DIR` ×10)
désignent des choses différentes selon l'étape et ne peuvent pas être fusionnés. Le gain est structurel.

## Points ouverts

1. Étape 10 : `kept` est **reconstruit** (cellule d'origine absente) ; assertion à 12 484 images.
2. Rôles encore à confirmer par exécution : 07a, 09b, 14, 16.
3. Chemins en dur `/content/wheat-rust-ai-pipeline` ; seul `PROJECT` est centralisé.
4. Exclus : utilitaires Drive/export, `archive/`, `diagnostics/`, `exploration/` (dans `benchmark_classification.zip`).
