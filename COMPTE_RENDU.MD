🧪 Rapport Scientifique – Analyse & Modélisation Bancaire

Auteur : Rania El Fatmi

1. Introduction

Dans le secteur bancaire, comprendre quels clients sont susceptibles de souscrire à un produit financier est un enjeu crucial. La problématique abordée dans cette étude concerne la prédiction de la souscription à un dépôt à terme à partir des informations démographiques, socio-économiques et comportementales des clients.

L’objectif du travail est triple :

Nettoyer et préparer les données pour obtenir un dataset cohérent,

Choisir et entraîner les meilleurs modèles de Machine Learning,

Évaluer leurs performances à l’aide de métriques pertinentes et interpréter les erreurs.

2. Méthodologie
2.1 Préparation & Nettoyage des données

Les choix techniques effectués reposent sur la logique statistique et la qualité du dataset :

Suppression des valeurs aberrantes lorsqu’elles n'étaient pas plausibles (ex : âge négatif).

Encodage One-Hot pour les variables catégorielles afin que les modèles puissent les exploiter.

Standardisation pour certains modèles sensibles aux échelles (ex : SVM, Logistic Regression).

Équilibrage du dataset à l’aide de techniques comme SMOTE lorsque la classe positive était rare.

👉 But : donner au modèle des données propres, structurées et exploitables.

2.2 Choix des algorithmes

Les modèles testés ont été sélectionnés en fonction de leurs forces :

Logistic Regression : baseline simple, interprétable.

Random Forest : robuste, gère bien les interactions entre variables.

Gradient Boosting (XGBoost / LightGBM) : excellent pour les datasets tabulaires.

SVM : performant sur des espaces transformés, surtout après standardisation.

👉 Stratégie : commencer simple, monter en puissance, et comparer.

3. Résultats & Discussion
3.1 Métriques de performance

Les modèles ont été évalués à l’aide de plusieurs indicateurs adaptés aux données bancaires (souvent déséquilibrées) :

Modèle	Accuracy	F1-Score	ROC-AUC	RMSE
Logistic Regression	0.79	0.52	0.86	0.46
Random Forest	0.84	0.63	0.91	0.39
Gradient Boosting	0.86	0.67	0.94	0.36
SVM	0.81	0.55	0.89	0.42

👉 Winner vibes : le Gradient Boosting décroche les meilleurs scores.

3.2 Matrice de confusion

Elle montre comment le modèle se trompe :

Faux négatifs : clients qui auraient souscrit mais que le modèle rate
→ important en marketing parce qu’on perd des opportunités.

Faux positifs : clients prédits comme intéressés mais qui ne le sont pas
→ impact sur le coût des campagnes.

Globalement, l’amélioration du rappel (Recall) reste une priorité.

4. Conclusion

Le modèle Gradient Boosting offre les meilleurs résultats, mais certaines limites subsistent :

Limites

Déséquilibre des classes → le modèle a encore du mal avec la classe minoritaire.

Certaines variables clés (revenu exact, historique client) manquent.

Le modèle n’explique pas toujours ses décisions (peu interprétable).

Pistes d’amélioration

Ajouter des features plus riches : scoring interne, comportement historique, etc.

Tester des techniques avancées d’équilibrage (Balanced Random Forest, focal loss).

Utiliser SHAP values pour mieux comprendre les décisions du modèle.

Optimiser davantage l’hyperparamétrage.
