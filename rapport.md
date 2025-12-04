# Compte Rendu : Dataset Bank Marketing
NOM ET PRENOM : RANIA EL FATMI 
## 1. Présentation Générale

"Bank Marketing" documente 45 211 appels téléphoniques d'une banque portugaise (2008-2010) pour promouvoir des dépôts à terme.

| Métrique | Valeur |
|----------|--------|
| **Instances** | 45 211 |
| **Variables** | 17 (16 features + 1 cible) |
| **Période** | Mai 2008 - Nov 2010 |
| **Objectif** | Prédire la souscription (yes/no) |
| **Valeurs manquantes** | 0% |

---

## 2. Structure des Données

### 2.1 Répartition par Type de Variables

```
Variables Numériques (6):
├─ age, balance, day, duration
├─ campaign, pdays, previous
└─ Range: continue ou discrète

Variables Catégorielles (10):
├─ job, marital, education, default
├─ housing, loan, contact, month
├─ poutcome
└─ Modalités: 2 à 12 catégories

Variable Cible (1):
└─ y (yes/no) - DÉSÉQUILIBRÉE
```

### 2.2 Distribution de la Variable Cible

| Classe | Pourcentage | Interprétation |
|--------|-------------|----------------|
| **No** (pas de souscription) | ~88% | Classe majoritaire |
| **Yes** (souscription) | ~12% | Classe minoritaire |

**Graphique de distribution:**
```
No  ████████████████████████████████████████ 88%
Yes █████ 12%
```

---

## 3. Catégories de Variables

### 3.1 Informations Client (8 variables)

| Variable | Type | Exemples | Impact |
|----------|------|----------|--------|
| age | Int | 18-95 | ⭐⭐⭐ |
| job | Cat | admin, technician, services | ⭐⭐⭐ |
| marital | Cat | married, single, divorced | ⭐⭐ |
| education | Cat | university, high.school | ⭐⭐⭐ |
| default | Bin | yes/no | ⭐⭐⭐⭐ |
| balance | Int | -8000 à +100000 EUR | ⭐⭐⭐⭐⭐ |
| housing | Bin | yes/no | ⭐⭐ |
| loan | Bin | yes/no | ⭐⭐ |

### 3.2 Informations Contact (4 variables)

| Variable | Type | Description | Note |
|----------|------|-------------|------|
| contact | Cat | cellular/telephone | Canal de communication |
| day | Int | 1-31 | Jour du contact |
| month | Cat | jan-dec | Mois du contact |
| duration | Int | Secondes | ⚠️ Disponible après appel |

### 3.3 Historique Campagnes (4 variables)

| Variable | Description | Valeur Spéciale |
|----------|-------------|-----------------|
| campaign | Nb contacts campagne actuelle | - |
| pdays | Jours depuis dernier contact | -1 = jamais contacté |
| previous | Nb contacts antérieurs | - |
| poutcome | Résultat campagne précédente | success/failure/other |

---

## 4. Statistiques Descriptives Clés

### 4.1 Variables Numériques - Tendances Centrales

| Variable | Moyenne | Médiane | Min | Max | Écart-type |
|----------|---------|---------|-----|-----|------------|
| age | ~40 ans | 39 | 18 | 95 | ~10 |
| balance | ~1400 EUR | 448 | -8019 | 102127 | ~3000 |
| campaign | ~2.5 | 2 | 1 | 63 | ~3 |
| duration | ~258 sec | 180 | 0 | 4918 | ~257 |

### 4.2 Top Catégories - Variables Catégorielles

**Distribution des professions (job):**
```
management       ████████████ 25%
blue-collar      ███████████ 22%
technician       ██████████ 19%
admin            ████████ 15%
services         ██████ 12%
autres           ███ 7%
```

**Distribution éducation:**
```
secondary        ████████████████ 51%
tertiary         ███████████ 30%
primary          ██████ 15%
unknown          █ 4%
```

---

## 5. Corrélations et Insights

### 5.1 Variables les Plus Prédictives

| Rang | Variable | Type de corrélation | Explication |
|------|----------|---------------------|-------------|
| 1 | **duration** | Très forte (+) | ⚠️ Clients intéressés parlent plus longtemps |
| 2 | **poutcome** | Forte (+) | Succès passé prédit succès futur |
| 3 | **balance** | Forte (+) | Capacité financière |
| 4 | **housing** | Moyenne (-) | Endettement réduit souscription |
| 5 | **contact** | Moyenne (+) | Cellular > telephone |

### 5.2 Patterns Temporels Observés

**Saisonnalité par mois:**
```
Jan-Mar:  Faible activité     ███
Apr-Jun:  Activité moyenne    ██████
Jul-Sep:  Forte activité      ██████████
Oct-Dec:  Pic de conversion   ████████████
```

**Effet du nombre de contacts:**
- 1-2 contacts: Taux de succès optimal
- 3-5 contacts: Décroissance
- 6+ contacts: Fortement négatif

---

## 6. Versions du Dataset

| Fichier | Instances | Features | Usage |
|---------|-----------|----------|-------|
| **bank-additional-full.csv** | 41 188 | 20 | ✅ Production |
| bank-additional.csv | 4 119 | 20 | Tests rapides |
| bank-full.csv | 41 188 | 17 | Legacy |
| bank.csv | 4 119 | 17 | Prototypage |

---

## 7. Points d'Attention Méthodologiques

### ⚠️ Problèmes Critiques

1. **Variable duration**: Ne peut pas être utilisée en prédiction pré-appel
2. **Déséquilibre des classes**: Nécessite techniques de rééquilibrage
3. **Ordre temporel**: Les versions full sont chronologiques (éviter data leakage)

### ✅ Forces du Dataset

1. **Aucune valeur manquante**
2. **Données réelles** d'une institution bancaire
3. **Période intéressante** (crise financière 2008-2009)
4. **Bien documenté** avec publication scientifique

---

## 8. Applications Recommandées

### Modèles de Classification

```
Approche 1: Baseline
└─ Logistic Regression (interprétabilité)

Approche 2: Performance
├─ Random Forest
├─ XGBoost / LightGBM
└─ Neural Networks

Approche 3: Ensemble
└─ Stacking de plusieurs modèles
```

### Métriques d'Évaluation

| Métrique | Usage | Priorité |
|----------|-------|----------|
| **AUC-ROC** | Capacité de discrimination globale | ⭐⭐⭐⭐⭐ |
| **F1-Score** | Balance précision/rappel | ⭐⭐⭐⭐⭐ |
| **Précision** | Minimiser faux positifs | ⭐⭐⭐⭐ |
| **Rappel** | Maximiser détection vrais positifs | ⭐⭐⭐⭐ |
| Accuracy | ⚠️ Biaisée par déséquilibre | ⭐⭐ |

---

## 9. Accès aux Données

### Code Python

```python
from ucimlrepo import fetch_ucirepo

# Téléchargement
bank_marketing = fetch_ucirepo(id=222)

# Extraction
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Info
print(bank_marketing.metadata)
```

### Licence
**CC BY 4.0** - Libre d'utilisation avec attribution

---

## 10. Résumé Visuel

```
┌─────────────────────────────────────────────────────┐
│           BANK MARKETING DATASET                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📊 45 211 clients                                   │
│  📞 Campagnes téléphoniques                         │
│  🎯 Objectif: Prédire souscription dépôt à terme   │
│                                                      │
│  STRUCTURE:                                          │
│  ├─ 8 variables client                              │
│  ├─ 4 variables contact                             │
│  ├─ 4 variables historique                          │
│  └─ 1 variable cible (yes/no)                       │
│                                                      │
│  QUALITÉ:                                            │
│  ✅ 0% valeurs manquantes                           │
│  ✅ Données réelles                                 │
│  ⚠️  Classes déséquilibrées (88% / 12%)            │
│                                                      │
│  UTILISATION:                                        │
│  → Classification binaire                           │
│  → Optimisation campagnes marketing                 │
│  → Scoring de leads                                 │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Conclusion

Bank Marketing offre une base solide pour développer des modèles prédictifs de marketing bancaire. Sa qualité exceptionnelle (pas de valeurs manquantes), sa taille substantielle (45k instances) et sa documentation complète en font une ressource de référence pour l'apprentissage automatique appliqué au domaine financier.
