# Adaptation de Domaine pour l'Analyse de Sentiments : Amazon vers Twitter
Ce projet explore les défis du **Domain Shift** dans l'analyse de sentiments en adaptant un modèle entraîné sur des critiques formelles (Amazon) vers des messages informels (Twitter) via l'**Apprentissage Actif**.

```text
PROJET/
├── code/
│   ├── main.py                    # Script principal 
│   ├── config.py                  # Paramètres et chemins 
│   ├── preprocessing.py           # Nettoyage, normalisation et lemmatisation
│   ├── active_learning_strategies.py # Implémentation des métriques 
│   ├── model_utils.py             # Pipeline Scikit-Learn et utilitaires modèle
│   ├── reporting_utils.py         # Gestion des logs et export des résultats
│   └── plot_comparison.py         # Génération du graphique comparatif final
├── dataset/
│   ├── Amazon_Fine/Reviews.csv    # Domaine Source
│   └── tweetEmotion/tweet_emotions.csv # Domaine Cible
└── results/                       # Logs (.txt) et courbes (.png) générés
```

---
### 1. Création de l'environnement 
```bash
# Création
python -m venv .venv

# Activation (Windows)
.venv\Scripts\activate

# Activation (macOS/Linux)
source .venv/bin/activate
```

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```
--- 

## Stratégies d'Incertitude
python main.py -o entropy
python main.py -o margin

## Stratégies de Diversité
python main.py -o density
python main.py -o diversity
python main.py -o max_dist

## Stratégies de Référence et Hybride
python main.py -o random
python main.py -o combined

## Génération du graphique final
python plot_comparison.py