#code/active_learning_strategies.py
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# --- STRATÉGIES BASÉES SUR L'INCERTITUDE (CONFIDENCE) ---

def get_entropy_scores(probs):
    """Calcule l'entropie de Shannon : plus elle est haute, plus le modèle est confus."""
    # On ajoute une petite valeur (1e-10) pour éviter le log(0)
    return -np.sum(probs * np.log2(probs + 1e-10), axis=1)

def get_margin_scores(probs):
    """Calcule la marge entre les deux classes les plus probables."""
    # On trie les probabilités par ordre croissant
    part = np.partition(probs, -2, axis=1)
    # Marge = Prob(classe_top1) - Prob(classe_top2)
    # Un score petit (marge faible) indique une grande incertitude
    margin = part[:, -1] - part[:, -2]
    # On renvoie 1 - marge pour que les scores les plus élevés soient les plus incertains
    return 1 - margin

# --- STRATÉGIES BASÉES SUR LA DIVERSITÉ ---

def get_k_means_representative_samples(X_vec, n_samples):
    """
    Sélectionne des échantillons représentatifs en utilisant K-Means.
    On prend les points les plus proches du centre de chaque cluster.
    """
    if n_samples >= X_vec.shape[0]:
        return np.arange(X_vec.shape[0])
    
    kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
    kmeans.fit(X_vec)
    
    # On trouve l'indice de l'échantillon le plus proche de chaque centre de cluster
    dists = pairwise_distances(kmeans.cluster_centers_, X_vec)
    indices = np.argmin(dists, axis=1)
    return np.unique(indices)

# --- MÉTHODE COMBINÉE (HYBRIDE) ---

def get_combined_scores(probs, entropy_weight=0.5, margin_weight=0.5):
    """Mélange deux scores d'incertitude (Entropie et Marge) pour une sélection robuste."""
    s_entropy = get_entropy_scores(probs)
    s_margin = get_margin_scores(probs)
    
    # Normalisation des scores entre 0 et 1 pour les combiner équitablement
    s_entropy /= (np.max(s_entropy) + 1e-10)
    s_margin /= (np.max(s_margin) + 1e-10)
    
    return (entropy_weight * s_entropy) + (margin_weight * s_margin)

def get_density_scores(X_vec):
    """
    Calcule la densité : plus le score est haut, plus l'échantillon 
    est représentatif du domaine cible (Twitter).
    """
    # Distance moyenne aux 10 plus proches voisins dans le pool
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X_vec)
    distances, _ = nn.kneighbors(X_vec)
    # Densité = inverse de la distance moyenne
    return 1.0 / (np.mean(distances, axis=1) + 1e-10)

def get_diverse_max_dist_indices(X_pool_vec, X_train_vec, n_to_add):
    """
    Sélectionne les échantillons les plus éloignés de ceux déjà appris.
    Répond au critère de diversité pure.
    """
    # Distance entre chaque point du pool et chaque point déjà annoté
    dists = pairwise_distances(X_pool_vec, X_train_vec, metric='euclidean')
    # Distance au voisin le plus proche déjà connu
    min_dists = np.min(dists, axis=1)
    # On prend les points les plus "isolés" (distance min maximale)
    return np.argsort(min_dists)[-n_to_add:]