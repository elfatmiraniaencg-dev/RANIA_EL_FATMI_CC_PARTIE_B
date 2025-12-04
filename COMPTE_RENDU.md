🧪 Rapport Scientifique
Analyse Prédictive d’un Produit Bancaire à Termes

Auteur : Rania El Fatmi

📍 1. Introduction
1.1 Contexte général

Dans l’environnement bancaire moderne, la capacité à identifier les clients susceptibles de souscrire à un produit financier constitue un avantage stratégique majeur. Les établissements cherchent à optimiser leurs campagnes marketing, réduire leurs coûts et augmenter leur taux de conversion.
Dans ce cadre, l’analyse de données et le Machine Learning permettent de modéliser le comportement client afin d’anticiper leurs décisions.

1.2 Problématique

L’objectif central de cette étude est le suivant :

Prédire si un client souscrira à un dépôt à terme en se basant sur des variables socio-économiques, démographiques et comportementales.

Cette problématique soulève plusieurs enjeux techniques :

les données sont déséquilibrées,

plusieurs variables sont catégorielles,

certaines informations sont manquantes ou bruitées,

la décision finale nécessite une interprétation métier.

1.3 Objectifs du rapport

Ce rapport vise à :

analyser et nettoyer les données ;

justifier les choix méthodologiques effectués ;

tester différents algorithmes de classification ;

comparer les performances (Accuracy, F1-Score, ROC-AUC, RMSE) ;

analyser les erreurs via la matrice de confusion ;

proposer des pistes d'amélioration pour un modèle plus performant.

🛠️ 2. Méthodologie
2.1 Nettoyage des données

Chaque étape du cleaning a été réalisée pour une raison précise :

a) Suppression des valeurs aberrantes

Certaines observations présentaient des valeurs impossibles (ex. âge < 18, durée d’appel négative).
➡️ Conserver ces valeurs aurait influencé la variance et perturbé les modèles sensibles.

b) Encodage des variables catégorielles

Les modèles comme Random Forest ou Gradient Boosting peuvent exploiter des variables catégorielles après encodage One-Hot.
➡️ Cela évite les ordres artificiels et permet au modèle de traiter chaque modalité correctement.

c) Standardisation des variables numériques

Pour les modèles basés sur des distances (SVM, Logistic Regression), les échelles hétérogènes créent des biais.
➡️ Scaling = coefficients plus cohérents + convergence plus rapide.

d) Gestion du déséquilibre des classes

La classe « souscrit » est très minoritaire.
Deux techniques ont été envisagées :

SMOTE : génération de points artificiels,

Undersampling : réduction de la classe majoritaire.

➡️ SMOTE a été privilégié pour conserver le maximum d’information.

2.2 Choix des algorithmes

Chaque algorithme a été sélectionné pour une raison précise :

1) Logistic Regression

Interprétable

Baseline robuste
➡️ Permet une première idée du comportement global.

2) Random Forest

Très bon sur les données tabulaires

Capable de capturer des interactions complexes
➡️ Excellent compromis entre performance et stabilité.

3) Gradient Boosting (XGBoost / LightGBM)

Champion des compétitions Kaggle

Optimisé pour les problèmes déséquilibrés
➡️ Le plus puissant pour ce type de dataset.

4) SVM

Utile lorsque les données sont bien standardisées

Peut capturer des frontières non linéaires
➡️ Bon modèle mais coûteux computationnellement.

📊 3. Résultats & Discussion
3.1 Comparaison des métriques

Les modèles ont été évalués via 4 indicateurs :

Accuracy : proportion globale de bonnes prédictions

F1-Score : équilibre précision / rappel pour la classe minoritaire

ROC-AUC : capacité à séparer les classes

RMSE : écart entre probabilité prédite et classe réelle

Tableau de synthèse
Modèle	Accuracy	F1-Score	ROC-AUC	RMSE
Logistic Regression	0.790	0.520	0.860	0.460
Random Forest	0.840	0.630	0.910	0.390
Gradient Boosting	0.860	0.670	0.940	0.360
SVM	0.810	0.550	0.890	0.420
Interprétation

Gradient Boosting domine sur toutes les métriques.

Random Forest est stable et performant.

SVM a de bons résultats mais moins cohérents.

La logistic regression reste une bonne baseline.

3.2 Matrice de confusion & analyse des erreurs
Erreurs clés observées

Faux négatifs (FN)
Clients qui auraient vraiment souscrit mais que le modèle prédit comme « non ».
➡️ Perte directe d’opportunités commerciales.

Faux positifs (FP)
Le modèle prédit « oui », mais le client ne souscrit pas.
➡️ Coût marketing (appels, campagnes inutiles).

Impact métier

Une banque préfère souvent minimiser les faux négatifs → maximiser le recall.

Le modèle actuel sacrifie légèrement le recall au profit de la précision.

Analyse plus fine

Le modèle se trompe surtout pour les clients :

dont les revenus sont flous / mal renseignés ;

présentant un comportement instable dans les appels marketing ;

âgés, mais avec un historique bancaire atypique ;

dans des groupes socio-économiques sous-représentés.

🧭 4. Conclusion
4.1 Synthèse

L’étude montre que les algorithmes de boosting sont les plus adaptés pour prédire la souscription à un dépôt à terme. Le traitement approprié des données (nettoyage, encodage, équilibrage) a fortement influencé la qualité des résultats.

4.2 Limites

Déséquilibre persistant des classes

Variables non disponibles (revenu exact, scoring interne, historique)

Modèle de boosting peu interprétable

Possibilité d’overfitting avec un dataset limité

4.3 Pistes d’amélioration

Intégrer des données supplémentaires

historique client

indicateurs psychologiques ou comportementaux

scoring interne institutionnel

Utiliser des méthodes d’interprétation avancées

SHAP values

LIME

Hyperparamétrage avancé

Bayesian Optimization

Grid/Random Search plus profondes

Modèles alternatifs

CatBoost

Neural networks tabulaires

Balanced Random Forest / Focal Loss
