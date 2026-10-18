# Correspondance notebook → scripts

| Étape | Notebook d'origine | Script |
|---|---|---|
| Inventaire et mapping de labels | 56–57 | `01_inventory_and_harmonize.py` |
| Validation de lisibilité | Les contrôles EDA sont principalement en 99–103 | `02_validate_images.py` |
| Doublons exacts SHA-256 | 56–59, 65 | `03_audit_exact_duplicates.py` |
| Audit perceptuel et décision humaine | 76–92 | Conserver les CSV de revue/décisions existants; ne pas automatiser/remplacer ces choix |
| Construction du dataset final | 93–97 | `04_build_classification_v3.py` |
| Validation des fichiers, SHA et partitions | 98 | `05_validate_classification_v3.py` |

Les scripts couvrent les opérations reproductibles, mais le registre des décisions de revue manuelle reste une entrée obligatoire pour reproduire exactement le benchmark final.
