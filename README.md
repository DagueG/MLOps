# Credit Scoring MLOps Project

Projet de scoring crédit avec une approche MLOps complète, du tracking des expériences à la pré-production du modèle.

## Objectif

Construire un modèle de classification automatisé pour prédire la probabilité qu'un client rembourse son crédit, avec :
- ✓ Optimisation des hyperparamètres
- ✓ Gestion du déséquilibre de classe
- ✓ Optimisation du seuil métier (minimisation du coût FN/FP)
- ✓ Transparency (SHAP, Feature Importance)
- ✓ MLOps (MLflow, Model Registry, Serving)

---

## Architecture du Projet

```
MLOps/
├── notebooks/              # Notebooks Jupyter pour chaque étape
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_hyperparameter_optimization.ipynb
│   └── 04_feature_importance_threshold.ipynb
├── src/                    # Code réutilisable
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── models/                 # Modèles sauvegardés
├── outputs/                # Données générées
│   ├── train_processed.csv
│   └── test_processed.csv
├── logs/                   # Logs MLflow
├── datasets/               # Données brutes
├── pyproject.toml          # Configuration uv
└── README.md              # Ce fichier
```

---

## Étapes du Projet

### ✅ Étape 1 - Préparation des Données
**Status:** Complétée

- Chargement et fusion de 8 sources de données
- Gestion des valeurs manquantes (imputation médiane/mode)
- Encodage des variables catégorielles
- Analyse du déséquilibre de classe (91.9% / 8.1%)

**Résultats:**
- Dataset d'entraînement: 307,511 × 148 colonnes
- Dataset de test: 48,744 × 121 colonnes
- Zéro valeurs manquantes

---

### ✅ Étape 2 - MLflow Tracking & Entraînement des Modèles
**Status:** Complétée

- Entraînement de 4 modèles baseline: Logistic Regression, Random Forest, XGBoost, LightGBM
- Validation croisée stratifiée (5-fold StratifiedKFold)
- Logging automatique avec MLflow
- Évaluation sur métriques techniques et métier (coût FN=10x, FP=1x)

**Résultats:**
| Modèle | AUC | Accuracy | Business Cost | CV AUC |
|--------|-----|----------|---------------|--------|
| LightGBM | 0.7657 | 0.7132 | 159,859 | 0.7657 |
| XGBoost | 0.7655 | 0.7153 | 159,658 | 0.7655 |
| Logistic Reg. | 0.7573 | 0.6950 | 163,281 | 0.7573 |
| Random Forest | 0.7413 | 0.7382 | 169,766 | 0.7413 |

---

### ⏳ Étape 3 - Optimisation des Hyperparamètres
**Status:** En cours

- GridSearchCV sur LightGBM et XGBoost
- Recherche d'hyperparamètres optimaux (n_estimators, max_depth, learning_rate)
- Logging des essais avec MLflow
- Comparaison baseline vs optimized

---

### ⏳ Étape 4 - Feature Importance & Optimisation du Seuil
**Status:** En cours

- Optimisation du seuil de décision basé sur coût métier
- Feature Importance par permutation
- Feature Importance globale avec SHAP (TreeExplainer)
- Sauvegarder le modèle final avec seuil optimal dans Model Registry

---

## Installation

### Prérequis
- Python >= 3.10
- uv (gestionnaire de paquets)

### Setup

```bash
# Installer les dépendances
uv sync

# Lancer un notebook
uv run jupyter notebook notebooks/01_data_preparation.ipynb

# Lancer MLflow (après Étape 2)
uv run mlflow ui
```

---

## Données

Les données proviennent du challenge Kaggle "Home Credit Default Risk".

**Tables principales:**
- `application_train.csv` - Données d'entraînement (307k clients)
- `application_test.csv` - Données de test (48k clients)
- `bureau.csv` - Historique crédit bureau
- `bureau_balance.csv` - Soldes mensuels bureau
- `credit_card_balance.csv` - Soldes cartes de crédit
- `installments_payments.csv` - Historique des versements
- `POS_CASH_balance.csv` - Soldes point de vente
- `previous_application.csv` - Applications antérieures

---

## Points Clés

⚠️ **Déséquilibre de classe:** 91.9% bons clients vs 8.1% mauvais
→ À gérer avec `class_weight='balanced'` ou SMOTE

📊 **Coût métier:** FN (mauvais client accepté) coûte 10x plus que FP (bon client refusé)
→ Optimiser le seuil selon ce ratio

---

## Auteur

DagueG - OpenClassrooms Project

---
