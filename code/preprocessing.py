#preprocessing.py
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|\@\w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s!]', '', text).lower()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if len(w) > 1])

def map_amazon_sentiment(score):
    if score <= 2: return 'NEGATIVE'
    elif score == 3: return 'NEUTRAL'
    return 'POSITIVE'