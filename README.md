#  Backend — API FastAPI & Intelligence Artificielle

Ce backend fournit les statistiques des Jeux Olympiques et permet de générer des prédictions de médailles à l’aide d’un modèle de Machine Learning (RandomForest).

---

##  Structure du Backend

backend/
├── app.py
├── model/
│   ├── olympics_rf_pack.joblib
│   └── country_medals_ml.csv
├── check_model.py
├── requirements.txt
└── .venv/

---

##  Dépendances

Créer un environnement virtuel puis installer les dépendances suivantes :

pip install fastapi uvicorn pandas numpy joblib
pip install scikit-learn==1.2.2

⚠️ Important :  
Le modèle a été entraîné avec scikit-learn version 1.2.2.  
Toute autre version peut provoquer des erreurs lors du chargement du modèle.

---

## ▶ Lancer le serveur

Depuis le dossier backend :

uvicorn app:app --reload

L’API sera accessible à l’adresse suivante :

http://127.0.0.1:8000

Documentation interactive (Swagger) :

http://127.0.0.1:8000/docs

---

## 🔗 Endpoints de l’API

### 🌍 Liste des pays
GET /countries

Réponse :
{
  "countries": ["France", "Morocco", "Japan", "..."]
}

---

###  Résumé des médailles
GET /medals/summary

Réponse :
[
  {
    "country": "France",
    "gold": 262,
    "silver": 289,
    "bronze": 330,
    "total": 881
  }
]

---

###  Prédiction de médailles
POST /predict

Requête :
{
  "country_name": "Morocco",
  "year": 2016
}

Réponse :
{
  "country_name": "Morocco",
  "year": 2016,
  "gold": 1,
  "silver": 0,
  "bronze": 0,
  "total": 1
}

---

##  Modèle de Machine Learning

Le modèle utilisé est un RandomForestRegressor entraîné sur des données historiques des Jeux Olympiques.

Trois modèles sont utilisés :
- Or (gold)
- Argent (silver)
- Bronze (bronze)

Variables d’entrée :
- Médailles passées
- Moyenne des 3 dernières olympiades
- Tendance (delta)

Sortie :
- Nombre prédit de médailles par type et total

---

## ❌ Base de données

Aucune base de données n’est utilisée.

Les données sont chargées depuis un fichier CSV et le modèle est chargé depuis un fichier joblib.

---

##  Vérification du modèle (optionnel)

Pour vérifier que le modèle se charge correctement :

python check_model.py

Sortie attendue :
Loaded model pack
KEYS: ['features', 'rf_gold', 'rf_silver', 'rf_bronze']

---

## ℹ Remarques

- Le modèle effectue une prédiction, il ne cherche pas une ligne exacte dans le CSV
- Une prédiction est possible même si l’année demandée n’existe pas dans les données
- Une erreur est renvoyée uniquement si le pays n’existe pas dans le dataset

---

##  Technologies utilisées

- Python 3.10
- FastAPI
- Pandas / NumPy
- Scikit-learn
- Joblib