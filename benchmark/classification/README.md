# Classification v3 — scripts de construction et d'audit

Ces scripts séparent les étapes de préparation du benchmark de l'entraînement du modèle. Ils reprennent les choix visibles dans le notebook, notamment le mapping de labels des cellules 56–57 et la reconstruction finale de `classification_v3` des cellules 92–98.

## Important : conserver les décisions manuelles
Le notebook a effectué des audits de doublons exacts et perceptuels, des revues visuelles et des décisions de quarantaine/KEEP. Il ne faut **pas** remplacer ces décisions par un nouveau split aléatoire. Le script de reconstruction utilise donc un **manifeste final approuvé** contenant `sha256`, `canonical_class`, `new_split` et au moins un chemin de source existant (`destination_path_v2`, `destination_path`, `image_path`, `path` ou `source`).

Le script `04_build_classification_v3.py` n'invente pas les décisions de revue. Il est en dry-run par défaut et écrit dans `classification_v3_rebuilt`, sans supprimer le dossier original.

## Fichiers
- `01_inventory_and_harmonize.py` : inventaire des images et application du mapping de labels explicite du notebook. Aucun fichier image n'est modifié.
- `02_validate_images.py` : vérifie que les images sont lisibles et exporte un rapport.
- `03_audit_exact_duplicates.py` : produit les groupes de doublons exacts SHA-256; aucune suppression automatique.
- `04_build_classification_v3.py` : reconstruit le dataset à partir du manifeste final approuvé, vérifie le SHA avant/après copie et garde les splits existants.
- `05_validate_classification_v3.py` : contrôle les fichiers et le manifeste.
- `configs/classification.yaml` : mapping/compteurs attendus issus du projet.

## Dépendances
```bash
pip install pandas pillow pyyaml
```

## Exécution sur Colab
Depuis `/content/wheat-rust-ai-pipeline` :

### 1. Inventaire des sources brutes
```bash
python benchmark/classification/scripts/01_inventory_and_harmonize.py \
  --raw-root data/raw \
  --output data/reports/classification_v3_build/raw_inventory.csv
```

### 2. Contrôle de lisibilité
```bash
python benchmark/classification/scripts/02_validate_images.py \
  --inventory data/reports/classification_v3_build/raw_inventory.csv
```

### 3. Audit des doublons exacts
```bash
python benchmark/classification/scripts/03_audit_exact_duplicates.py \
  --inventory data/reports/classification_v3_build/raw_inventory.csv
```

### 4. Reconstruire à partir du manifeste final validé
Utiliser le CSV final conservé depuis le notebook (la table `kept` exportée à la fin de la revue, ou le manifeste final équivalent). Ne pas utiliser un simple inventaire brut.

D'abord simuler :
```bash
python benchmark/classification/scripts/04_build_classification_v3.py \
  --approved-manifest data/processed/classification_v2/approved_kept_manifest.csv \
  --output-dir data/processed/classification_v3_rebuilt
```

Si les chemins de source, classes, partitions et hashes sont corrects, lancer la copie :
```bash
python benchmark/classification/scripts/04_build_classification_v3.py \
  --approved-manifest data/processed/classification_v2/approved_kept_manifest.csv \
  --output-dir data/processed/classification_v3_rebuilt --execute
```

`approved_kept_manifest.csv` est un nom d'exemple : remplacez-le par le chemin réel du manifeste final sauvegardé dans votre Drive/Colab. Les noms des fichiers de registre dépendent de ce qui a été exporté pendant la revue manuelle.

### 5. Vérifier
```bash
python benchmark/classification/scripts/05_validate_classification_v3.py \
  --root data/processed/classification_v3_rebuilt
```
Pour exiger les comptes de référence exacts (uniquement si l'on reconstruit la version de référence) : ajouter `--expect-12484`.

## Référence du notebook
- Cellules 56–59 : inventaire global, harmonisation et audits SHA-256.
- Cellules 65–73 : manifeste global, déduplication et premiers splits.
- Cellules 76–92 : audit pHash, revue visuelle, quarantaines et manifeste final approuvé.
- Cellules 93–98 : chemins uniques, copie finale et validation de `classification_v3`.

Les scripts ont été vérifiés syntaxiquement, mais n'ont pas été exécutés sur les données réelles dans cette génération. Le benchmark ne doit pas être entraîné avant que la validation finale passe.
