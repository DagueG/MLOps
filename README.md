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
│   ├── 02_eda_and_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_optimization.ipynb
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

### ⏳ Étape 2 - MLflow Tracking & Exploration
**Status:** À faire

- Mise en place du tracking MLflow
- Lancement de l'interface web MLflow
- Configuration du Model Registry

---

### ⏳ Étape 3 - Entraînement & Comparaison des Modèles
**Status:** À faire

- Entraînement de plusieurs modèles (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Validation croisée stratifiée
- Évaluation sur des métriques métier (coût FN/FP) et techniques (AUC, Accuracy)

---

### ⏳ Étape 4 - Optimisation & Feature Importance
**Status:** À faire

- Optimisation des hyperparamètres (GridSearchCV/Optuna)
- Optimisation du seuil de décision
- Feature Importance globale (SHAP) et locale
- Sélection du meilleur modèle

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

🎯 **Benchmark:** Kaggle winner AUC = 0.82
→ Attention à l'overfitting si AUC > 0.82

---

## Auteur

Data Scientist - OpenClassrooms Project

---

## Statut du Projet

🟡 En cours - Étape 1 complétée
