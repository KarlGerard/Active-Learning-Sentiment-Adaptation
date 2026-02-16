# code/model_utils.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_model_pipeline(max_features, ngram_range):
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english', min_df=3)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs'))
    ])

def get_model_params_count(model):
    """Calcule le nombre de paramètres du modèle."""
    clf = model.named_steps['clf']
    return {
        "coef": clf.coef_.size,
        "intercept": clf.intercept_.size,
        "total": clf.coef_.size + clf.intercept_.size
    }

def plot_confusion_matrix(y_true, y_pred, title, cmap='Blues'):
    labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.show()