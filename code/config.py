# code/config.py

# Chemins des fichiers (relatifs au dossier 'code')
PATH_AMAZON = '../dataset/Amazon_Fine/Reviews.csv'
PATH_TWITTER = '../dataset/tweetEmotion/tweet_emotions.csv'
OUTPUT_DIR = '../results/' 

# Mapping des sentiments Twitter
TW_MAPPING = {
    'happiness': 'POSITIVE', 'love': 'POSITIVE', 'fun': 'POSITIVE',
    'enthusiasm': 'POSITIVE', 'relief': 'POSITIVE',
    'neutral': 'NEUTRAL',
    'sadness': 'NEGATIVE', 'empty': 'NEGATIVE', 'boredom': 'NEGATIVE',
    'worry': 'NEGATIVE', 'hate': 'NEGATIVE', 'anger': 'NEGATIVE'
}

# Paramètres de l'expérience
N_ITERATIONS = 5
BUDGETS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

# Configuration TF-IDF et Modèle
MAX_FEATURES = 15000
NGRAM_RANGE = (1, 2)