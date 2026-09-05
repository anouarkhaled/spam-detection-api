---
title: Spam Detection API
emoji: 🛡️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---

# Spam Detection API

Un classifieur de spam SMS emballé comme un vrai service : API + tests + conteneur Docker + intégration continue + monitoring — pas juste un notebook avec un score.

**Démo :** [lien Hugging Face Space à ajouter après déploiement]

## Le problème

Filtrer les messages indésirables (spam) des messages légitimes (ham), et le faire comme un service qu'on peut appeler depuis une vraie application — pas comme une cellule de notebook isolée.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (Almeida & Gómez Hidalgo) — 5 572 SMS étiquetés `spam`/`ham`, dataset académique classique, largement utilisé pour ce type de tâche.

- 4 825 ham, 747 spam (classes déséquilibrées ~13% spam)
- Split train/test stratifié 80/20, `random_state=42`

## Résultats

Deux modèles comparés (`src/train.py`) sur TF-IDF (1-2 grams) + un classifieur :

| Modèle | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.976** | **0.896** | **0.926** | **0.911** |
| Naive Bayes | 0.970 | 1.000 | 0.779 | 0.875 |

Logistic Regression est retenu (meilleur F1). Matrice de confusion (test, n=1115) :

| | Prédit ham | Prédit spam |
|---|---:|---:|
| **Réel ham** | 950 | 16 |
| **Réel spam** | 11 | 138 |

Naive Bayes a une precision parfaite mais rate 1 spam sur 5 (recall 0.78) — pour un filtre anti-spam, laisser passer du spam est généralement pire qu'un faux positif occasionnel, d'où le choix de Logistic Regression malgré une precision légèrement inférieure.

## Architecture

```
Client → POST /predict → TF-IDF + Logistic Regression (chargé une fois au démarrage) → {label, spam_probability}
                       ↘ enregistré en mémoire → GET /stats, GET /dashboard
```

- **Entraînement** (`src/train.py`) : TF-IDF (bi-grammes, stop-words anglais) + Logistic Regression, sauvegarde du pipeline complet (`model/model.joblib`) et des métriques (`model/metrics.json`).
- **Service** (`app/main.py`, FastAPI) :
  - `POST /predict` — `{"text": "..."}` → `{"label": "spam"|"ham", "spam_probability": 0.0-1.0}`
  - `GET /stats` — compteurs agrégés + 50 dernières prédictions (monitoring en mémoire)
  - `GET /dashboard` — page HTML avec graphique en direct (Chart.js) et un formulaire de test
  - `GET /health` — vérification de disponibilité
- **Conteneur** : le modèle est entraîné **au moment du build Docker** (pas de binaire committé) — l'image est donc reproductible à partir du seul code source.
- **CI** (`.github/workflows/ci.yml`) : à chaque push/PR sur `main` — installation, entraînement, tests (`pytest`), puis vérification que l'image Docker se construit.

## Installation locale

```bash
pip install -r requirements-dev.txt
python src/train.py --data data/sms.tsv --out model
pytest -q
uvicorn app.main:app --reload
```

Puis ouvrir `http://localhost:8000/dashboard`.

## Avec Docker

```bash
docker build -t spam-detection-api .
docker run -p 7860:7860 spam-detection-api
```

## Déploiement sur Hugging Face Spaces

1. Créer un compte sur [huggingface.co](https://huggingface.co) si besoin, puis un nouveau **Space** (type **Docker**).
2. Cloner le dépôt du Space créé, y copier le contenu de ce repo (le `README.md` a déjà l'en-tête YAML attendu par HF Spaces en haut de ce fichier).
3. `git push` vers le remote du Space — HF construit l'image Docker automatiquement et expose le service sur le port 7860.

## Limites actuelles

- Le monitoring (`/stats`) est **en mémoire** : il repart à zéro à chaque redémarrage du conteneur. En production, ce serait une vraie base de séries temporelles (Prometheus, TimescaleDB...).
- Modèle TF-IDF + régression logistique : rapide et déjà à 97.6% d'accuracy sur ce dataset, mais un modèle de langage (DistilBERT fine-tuné) capturerait mieux les tentatives de spam plus subtiles — non fait ici pour garder le service léger et rapide à démarrer.
- Pas de détection de dérive (data drift) sur le texte entrant — un futur ajout logique du monitoring actuel.
- Le build Docker n'a pas pu être testé de bout en bout dans cet environnement (Docker Desktop non démarré) — l'API elle-même, elle, a été testée en local (tests automatisés + appels manuels).

## Ce que j'ai appris

- Sur des classes déséquilibrées (13% de spam), l'accuracy seule masque le vrai comportement du modèle — c'est la comparaison precision/recall par classe qui a fait pencher le choix vers Logistic Regression plutôt que Naive Bayes (meilleur score brut mais recall spam trop faible).
- Entraîner le modèle **au moment du build Docker** plutôt que de committer un fichier `.joblib` évite un piège classique : un binaire de modèle qui se désynchronise silencieusement du code qui l'a produit.
- Séparer clairement "modèle" (`src/`), "service" (`app/`) et "infrastructure" (`Dockerfile`, CI) rend chaque partie testable isolément — les tests de `src/train.py` ne dépendent pas de FastAPI, et les tests de l'API ne re-testent pas la qualité du modèle.

## Stack

Python · scikit-learn · FastAPI · Docker · GitHub Actions · Chart.js
