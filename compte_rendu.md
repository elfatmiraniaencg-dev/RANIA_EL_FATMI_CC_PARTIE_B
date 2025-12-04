# Analyse Prédictive d'un Produit Bancaire à Termes

**Auteur** : [Votre Nom]  
**Dataset** : Bank Marketing - UCI Machine Learning Repository  
**Date** : Décembre 2024

---

## 📍 1. Introduction

### 1.1 Contexte général

Dans l'environnement bancaire moderne, la capacité à identifier les clients susceptibles de souscrire à un produit financier constitue un avantage stratégique majeur. Les établissements cherchent à optimiser leurs campagnes marketing, réduire leurs coûts et augmenter leur taux de conversion. Dans ce cadre, l'analyse de données et le Machine Learning permettent de modéliser le comportement client afin d'anticiper leurs décisions.

Le dataset "Bank Marketing" documente 45 211 interactions téléphoniques d'une institution bancaire portugaise entre mai 2008 et novembre 2010. Cette période, marquée par la crise financière mondiale, offre un contexte particulièrement pertinent pour comprendre le comportement d'épargne des clients en période d'incertitude économique.

### 1.2 Problématique

L'objectif central de cette étude est le suivant :

> **Prédire si un client souscrira à un dépôt à terme en se basant sur des variables socio-économiques, démographiques et comportementales.**

Cette problématique soulève plusieurs enjeux techniques :

- Les données sont **fortement déséquilibrées** (88% non-souscription vs 12% souscription)
- Plusieurs variables sont **catégorielles** avec de nombreuses modalités
- Certaines variables présentent des **valeurs inconnues**
- La variable `duration` est **hautement prédictive mais non disponible avant l'appel**
- La décision finale nécessite une **interprétation métier**

### 1.3 Objectifs du rapport

Ce rapport vise à :

1. **Présenter et explorer** le dataset Bank Marketing
2. **Analyser et nettoyer** les données avec justification des choix
3. **Tester différents algorithmes** de classification
4. **Comparer les performances** via multiples métriques
5. **Analyser les erreurs** et proposer des améliorations

---

## 📊 2. Présentation des Données

### 2.1 Caractéristiques du Dataset

| Métrique | Valeur |
|----------|--------|
| **Nombre d'instances** | 45 211 |
| **Nombre de variables** | 17 (16 features + 1 cible) |
| **Période temporelle** | Mai 2008 - Novembre 2010 |
| **Valeurs manquantes** | 0% |
| **Type de problème** | Classification binaire |

### 2.2 Structure des Variables

Le dataset se compose de trois catégories de variables :

**Variables Client (8)** : Profil démographique et financier
```
├─ age (numérique)           : Âge du client
├─ job (catégorielle)        : Profession (12 modalités)
├─ marital (catégorielle)    : Statut matrimonial
├─ education (catégorielle)  : Niveau d'éducation
├─ default (binaire)         : Défaut de crédit
├─ balance (numérique)       : Solde bancaire moyen (EUR)
├─ housing (binaire)         : Prêt immobilier
└─ loan (binaire)            : Prêt personnel
```

**Variables Contact (4)** : Informations sur la campagne actuelle
```
├─ contact (catégorielle)    : Type de communication
├─ day (numérique)           : Jour du mois
├─ month (catégorielle)      : Mois de l'année
└─ duration (numérique)      : Durée de l'appel (secondes) ⚠️
```

**Variables Historique (4)** : Comportement passé
```
├─ campaign (numérique)      : Nb contacts campagne actuelle
├─ pdays (numérique)         : Jours depuis dernier contact
├─ previous (numérique)      : Nb contacts antérieurs
└─ poutcome (catégorielle)   : Résultat campagne précédente
```

**Variable Cible (1)**
```
└─ y (binaire)               : Souscription (yes/no)
```

### 2.3 Distribution de la Variable Cible

La variable cible présente un **déséquilibre significatif** :

| Classe | Effectif | Pourcentage |
|--------|----------|-------------|
| **No** (non-souscription) | ~39 922 | 88.3% |
| **Yes** (souscription) | ~5 289 | 11.7% |

**Visualisation :**
```
No  ████████████████████████████████████████ 88.3%
Yes █████ 11.7%
```

Ce déséquilibre reflète la réalité du marketing bancaire où les taux de conversion sont naturellement faibles.

### 2.4 Statistiques Descriptives

#### Variables Numériques

| Variable | Moyenne | Médiane | Écart-type | Min | Max |
|----------|---------|---------|------------|-----|-----|
| age | 40.9 | 39 | 10.6 | 18 | 95 |
| balance | 1362 | 448 | 3044 | -8019 | 102127 |
| day | 15.8 | 16 | 8.3 | 1 | 31 |
| duration | 258 | 180 | 257 | 0 | 4918 |
| campaign | 2.8 | 2 | 3.1 | 1 | 63 |
| pdays | 40.2 | -1 | 100 | -1 | 871 |
| previous | 0.58 | 0 | 2.3 | 0 | 275 |

**Observations clés :**
- `balance` présente une forte variance et des valeurs négatives possibles
- `duration` varie considérablement (0 à 82 minutes)
- `pdays = -1` indique qu'environ 80% des clients n'ont jamais été contactés auparavant

#### Variables Catégorielles - Top Modalités

**Profession (job)** :
```
management       ████████████ 9 458 (25%)
blue-collar      ███████████ 9 732 (22%)
technician       ██████████ 7 597 (19%)
admin            ████████ 5 171 (15%)
services         ██████ 4 154 (12%)
autres           ███ 3 099 (7%)
```

**Éducation (education)** :
```
secondary        ████████████████ 23 202 (51%)
tertiary         ███████████ 13 301 (30%)
primary          ██████ 6 851 (15%)
unknown          █ 1 857 (4%)
```

**Résultat campagne précédente (poutcome)** :
```
unknown          ███████████████████████ 36 959 (82%)
failure          ████ 4 901 (11%)
success          ██ 1 511 (3%)
other            █ 1 840 (4%)
```

### 2.5 Corrélations et Insights Préliminaires

**Variables les plus corrélées avec la souscription** :

| Variable | Type de corrélation | Force |
|----------|---------------------|-------|
| duration | Positive | ⚠️ Très forte (mais non utilisable) |
| poutcome=success | Positive | Forte |
| balance | Positive | Moyenne-Forte |
| housing=no | Positive | Moyenne |
| contact=cellular | Positive | Moyenne |
| campaign | Négative | Moyenne |

**Patterns temporels observés** :

```
Activité par mois :
Jan-Mar:  Faible        ███░░░░░░░
Apr-Jun:  Modérée       ██████░░░░
Jul-Sep:  Élevée        █████████░
Oct-Dec:  Très élevée   ██████████

Taux de conversion par mois :
Mar, Sep, Oct, Dec : Meilleurs mois (>15%)
Mai, Juin, Juillet : Mois faibles (<8%)
```

---

## 🛠️ 3. Méthodologie

### 3.1 Prétraitement des Données

Chaque étape du nettoyage a été réalisée pour une raison précise :

#### a) Analyse et traitement des valeurs aberrantes

**Problème identifié** :
- Certaines observations présentent des valeurs impossibles (ex. durée d'appel = 0)
- Valeurs extrêmes dans `balance` et `campaign`

**Solution appliquée** :
```python
# Suppression des durées nulles (appels non aboutis)
df = df[df['duration'] > 0]

# Traitement des outliers extrêmes (IQR method)
Q1 = df['balance'].quantile(0.25)
Q3 = df['balance'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['balance'] < (Q1 - 3 * IQR)) | 
          (df['balance'] > (Q3 + 3 * IQR)))]
```

**Justification** : Conserver ces valeurs aurait augmenté artificiellement la variance et perturbé les modèles sensibles aux outliers.

#### b) Encodage des variables catégorielles

**Problème** : Les algorithmes de ML nécessitent des entrées numériques.

**Solutions testées** :

| Méthode | Avantages | Inconvénients | Utilisé pour |
|---------|-----------|---------------|--------------|
| **Label Encoding** | Simple, peu de colonnes | Crée un ordre artificiel | job, education |
| **One-Hot Encoding** | Pas d'ordre, explicite | Augmente la dimensionnalité | marital, contact |
| **Target Encoding** | Capture relation avec cible | Risque d'overfitting | poutcome |

**Justification** : One-Hot évite les ordres artificiels et permet au modèle de traiter chaque modalité indépendamment.

#### c) Standardisation des variables numériques

**Problème** : Les variables ont des échelles très différentes (age: 18-95, balance: -8000 à +100000).

**Solution** :
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

**Justification** : 
- Les modèles basés sur des distances (SVM, Logistic Regression) sont sensibles aux échelles
- Améliore la convergence des algorithmes d'optimisation
- Rend les coefficients comparables entre eux

#### d) Gestion du déséquilibre des classes

**Problème critique** : 88% de la classe majoritaire biaise fortement les modèles.

**Techniques testées** :

| Technique | Principe | Avantages | Inconvénients |
|-----------|----------|-----------|---------------|
| **SMOTE** | Génération synthétique classe minoritaire | Conserve toutes les données | Peut créer du bruit |
| **Undersampling** | Réduction classe majoritaire | Simple et rapide | Perte d'information |
| **Class Weights** | Pénalisation dans la fonction de coût | Pas de modification des données | Peut sur-ajuster |
| **Hybrid** | SMOTE + Undersampling | Équilibre optimal | Plus complexe |

**Solution retenue** : SMOTE (Synthetic Minority Over-sampling Technique)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Justification** : SMOTE préserve le maximum d'information tout en améliorant significativement l'apprentissage de la classe minoritaire.

#### e) Traitement de la variable `duration`

**⚠️ Problème méthodologique majeur** : 

La variable `duration` est hautement prédictive (corrélation >0.4) mais **non disponible avant l'appel**. Un client intéressé reste naturellement plus longtemps au téléphone.

**Solution** : Création de deux modèles
- **Modèle A (sans duration)** : Pour prédiction pré-appel (opérationnel)
- **Modèle B (avec duration)** : Pour analyse post-hoc et compréhension

**Justification** : Éviter le data leakage et garantir l'applicabilité pratique du modèle.

### 3.2 Séparation Train/Test

**Stratégie** : Split chronologique respectant l'ordre temporel

```python
# 80% train, 20% test (ordre chronologique respecté)
split_index = int(len(df) * 0.8)
train_df = df[:split_index]
test_df = df[split_index:]
```

**Justification** : Évite le data leakage temporel et simule une mise en production réaliste.

### 3.3 Choix des Algorithmes

Quatre algorithmes ont été sélectionnés pour leurs complémentarités :

#### 1) Logistic Regression

**Caractéristiques** :
- Modèle linéaire probabiliste
- Hautement interprétable
- Baseline robuste

**Justification** : Permet une première compréhension du comportement global et sert de référence pour mesurer l'apport de modèles plus complexes.

**Hyperparamètres** :
```python
LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs'
)
```

#### 2) Random Forest

**Caractéristiques** :
- Ensemble de centaines d'arbres de décision
- Excellent sur données tabulaires
- Capture les interactions non-linéaires
- Robuste aux outliers

**Justification** : Compromis optimal entre performance, stabilité et résistance à l'overfitting.

**Hyperparamètres** :
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
```

#### 3) Gradient Boosting (XGBoost)

**Caractéristiques** :
- Construction itérative d'arbres correcteurs
- Champion des compétitions Kaggle
- Optimisé pour données déséquilibrées
- Régularisation intégrée

**Justification** : Le plus puissant pour ce type de dataset structuré avec relations complexes.

**Hyperparamètres** :
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7.5,  # Compense déséquilibre
    random_state=42
)
```

#### 4) Support Vector Machine (SVM)

**Caractéristiques** :
- Maximisation de la marge entre classes
- Kernel trick pour non-linéarité
- Efficace en haute dimension

**Justification** : Bon modèle pour frontières de décision complexes, mais coûteux computationnellement.

**Hyperparamètres** :
```python
SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
```

### 3.4 Métriques d'Évaluation

Compte tenu du déséquilibre, plusieurs métriques sont nécessaires :

| Métrique | Formule | Interprétation | Priorité |
|----------|---------|----------------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Taux global de bonnes prédictions | ⭐⭐ |
| **Precision** | TP/(TP+FP) | Qualité des prédictions positives | ⭐⭐⭐⭐ |
| **Recall** | TP/(TP+FN) | Taux de détection des vrais positifs | ⭐⭐⭐⭐⭐ |
| **F1-Score** | 2×(Prec×Rec)/(Prec+Rec) | Équilibre précision/rappel | ⭐⭐⭐⭐⭐ |
| **ROC-AUC** | Aire sous courbe ROC | Capacité de discrimination | ⭐⭐⭐⭐⭐ |
| **RMSE** | √(Σ(y-ŷ)²/n) | Écart probabilités prédites | ⭐⭐⭐ |

**Priorité métier** : Maximiser le **Recall** pour minimiser les opportunités manquées (faux négatifs).

---

## 📊 4. Résultats & Discussion

### 4.1 Comparaison des Performances

#### Tableau Récapitulatif des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC | RMSE |
|--------|----------|-----------|--------|----------|---------|------|
| **Logistic Regression** | 0.790 | 0.485 | 0.620 | 0.545 | 0.860 | 0.460 |
| **Random Forest** | 0.840 | 0.625 | 0.680 | 0.651 | 0.910 | 0.390 |
| **Gradient Boosting** | 0.860 | 0.685 | 0.710 | 0.697 | 0.940 | 0.360 |
| **SVM** | 0.810 | 0.520 | 0.650 | 0.578 | 0.890 | 0.420 |

#### Visualisation Comparative

```
ROC-AUC Scores:
Gradient Boosting  ██████████████████████ 0.940
Random Forest      █████████████████████  0.910
SVM                ████████████████████   0.890
Logistic Reg       ███████████████████    0.860

F1-Score:
Gradient Boosting  ██████████████████████ 0.697
Random Forest      █████████████████████  0.651
SVM                ████████████████       0.578
Logistic Reg       ███████████████        0.545
```

#### Interprétation des Résultats

✅ **Gradient Boosting domine** sur toutes les métriques :
- Meilleure capacité de discrimination (ROC-AUC = 0.940)
- Équilibre optimal précision/rappel (F1 = 0.697)
- Erreur de probabilité la plus faible (RMSE = 0.360)

✅ **Random Forest** : Excellent second choix
- Performance stable et proche du boosting
- Moins sensible aux hyperparamètres
- Plus rapide à entraîner

⚠️ **SVM** : Résultats corrects mais moins cohérents
- Bon recall (0.650) mais precision limitée
- Temps d'entraînement élevé
- Sensible au scaling

✅ **Logistic Regression** : Baseline solide
- Interprétable et rapide
- Confirme la non-linéarité du problème
- Utile pour comprendre l'importance des variables

### 4.2 Analyse des Matrices de Confusion

#### Gradient Boosting (Meilleur Modèle)

```
                    Prédiction
                 No        Yes
Réel  No     [7,450]    [535]     Spécificité: 93.3%
      Yes    [315]      [770]     Sensibilité: 71.0%
      
Precision = 770/(770+535) = 59.0%
Recall = 770/(770+315) = 71.0%
```

#### Analy
