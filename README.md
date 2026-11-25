# Analyse des sentiments tunisiens

Projet de traitement du langage naturel (NLP) visant à classer des commentaires tunisiens écrits en **Arabizi** en deux catégories : **positif** (`1`) ou **négatif** (`-1`).

**Auteur : Mohamed Gharbi — étudiant en 4e année à ESPRIT, Data Analytics & Data Science.**

## État du projet

Code préparé pour l'entraînement, l'évaluation et la prédiction en ligne de commande. **L'entraînement sur TUNIZI et les performances réelles restent à valider. Aucun score n'est annoncé.**

Le dataset, les modèles entraînés et les rapports ne sont pas inclus dans ce dépôt.

## Technologies

- Python 3.10 ou supérieur
- pandas : lecture et nettoyage des données
- scikit-learn : TF-IDF, régression logistique et évaluation
- joblib : sauvegarde du pipeline entraîné

## Méthode

```text
Commentaires Arabizi
        |
Nettoyage + sélection des labels -1 / 1
        |
Suppression des doublons et annotations contradictoires
        |
Découpage stratifié : 80 % train / 20 % test
        |
TF-IDF : groupes de 3 à 5 caractères
        |
Régression logistique
        |
Évaluation sur le test : accuracy, précision, rappel, F1
```

Les groupes de caractères permettent de représenter les variantes d'écriture fréquentes en Arabizi. Le vocabulaire TF-IDF est appris **uniquement sur les données d'entraînement**, afin de limiter les fuites entre train et test. Le découpage utilise `random_state=42`.

## Installation

Depuis le dossier du projet :

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sous Linux ou macOS, utilisez `.venv/bin/python` à la place de `.venv\Scripts\python.exe`.

## Données attendues

Obtenez le dataset **TUNIZI** auprès de sa source autorisée et vérifiez ses conditions d'utilisation. Placez votre fichier dans `data/TuniziDataset.csv`.

Le CSV doit contenir les colonnes suivantes :

| Colonne | Contenu |
| --- | --- |
| `InputText` | Commentaire en Arabizi |
| `SentimentLabel` | `1` positif, `-1` négatif, `0` neutre |

Les commentaires neutres, les textes vides et les labels invalides sont exclus. Les textes identiques portant des étiquettes contradictoires sont retirés avant déduplication. Le nombre effectif d'exemples après nettoyage est écrit dans le rapport.

Le dossier `data/` et les CSV sont ignorés par Git : aucune redistribution du dataset n'est prévue.

## Entraîner et évaluer

```powershell
.venv\Scripts\python.exe sentiment.py train --data data/TuniziDataset.csv
```

La commande affiche les métriques effectivement calculées et crée :

- `models/sentiment.joblib` : pipeline TF-IDF + modèle entraîné
- `reports/metrics.json` : tailles des jeux de données, accuracy, rapport de classification et matrice de confusion

La matrice de confusion utilise l'ordre `[-1, 1]` ; les lignes représentent les classes réelles et les colonnes les prédictions.

Pour modifier la fraction de test :

```powershell
.venv\Scripts\python.exe sentiment.py train --data data/TuniziDataset.csv --test-size 0.25
```

## Prédire sur de nouveaux commentaires

Après entraînement :

```powershell
.venv\Scripts\python.exe sentiment.py predict --text "3ejbetni l kahwa" "service khayeb barcha"
```

Ces phrases sont des exemples de saisie. Leurs prédictions ne sont pas préremplies : elles seront calculées par votre modèle entraîné.

Chargez uniquement un modèle joblib que vous avez créé ou dont vous connaissez la provenance.

## Vérifications du nettoyage

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ces tests portent sur le prétraitement ; ils ne mesurent pas la qualité du modèle sur TUNIZI.

## Limites

- Classification binaire : aucune classe neutre n'est reconnue à la prédiction.
- Une phrase très courte ou sans contexte recevra tout de même une classe.
- Sarcasme, mélange des langues et variations d'écriture peuvent produire des erreurs.
- Le dédoublonnage exact ne supprime pas tous les commentaires presque identiques.
- Les scores sur un seul découpage ne suffisent pas à conclure à une robustesse en production.

## Prochaines étapes

- Exécuter l'entraînement sur le CSV et analyser les erreurs.
- Comparer différentes représentations TF-IDF et valider sur plusieurs découpages.
- Évaluer une version prenant en compte la classe neutre.

## Contact

[gharbi.mohamed.1@esprit.tn](mailto:gharbi.mohamed.1@esprit.tn)
