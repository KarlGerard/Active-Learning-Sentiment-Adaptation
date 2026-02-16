import argparse
import pandas as pd
import numpy as np
import copy, csv, time, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config
from preprocessing import clean_text, map_amazon_sentiment
from model_utils import create_model_pipeline, get_model_params_count
from reporting_utils import ResultsManager
import active_learning_strategies as al_strat

def main():
    # --- GESTION DES ARGUMENTS ---
    parser = argparse.ArgumentParser(description='Active Learning Sentiment Analysis')
    parser.add_argument('-o', '--strategy', type=str, default='random',
                        choices=['random', 'entropy', 'margin', 'diversity', 'density', 'max_dist', 'combined'],
                        help='Stratégie de sélection')
    args = parser.parse_args()
    
    strategy = args.strategy
    rm = ResultsManager(config.OUTPUT_DIR, strategy_name=strategy)

    # --- 1. CHARGEMENT & PRÉPARATION AMAZON (DOMAIN A) ---
    rm.log("--- Phase 1 : Préparation du Domaine Source (Amazon) ---")
    df_az = pd.read_csv(config.PATH_AMAZON, usecols=['Score', 'Text'], engine='python', 
                        on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL).sample(40000, random_state=42)
    df_az['sentiment'] = df_az['Score'].apply(map_amazon_sentiment)
    df_az['text_clean'] = df_az['Text'].apply(clean_text)
    
    X_train_az, X_test_az, y_train_az, y_test_az = train_test_split(
        df_az['text_clean'], df_az['sentiment'], test_size=0.2, random_state=42, stratify=df_az['sentiment'])

    # --- 2. ENTRAÎNEMENT & PERFORMANCE BASELINE (Point 16 & 17 du Projet) ---
    model_A = create_model_pipeline(config.MAX_FEATURES, config.NGRAM_RANGE)
    
    rm.log("\nEntraînement de la Baseline sur Amazon...")
    start_train_base = time.time()
    model_A.fit(X_train_az, y_train_az)
    duration_base = time.time() - start_train_base
    
    # Évaluation Baseline sur Domaine A (Amazon)
    acc_az = accuracy_score(y_test_az, model_A.predict(X_test_az))
    params = get_model_params_count(model_A)
    
    rm.log(f"Temps d'entraînement Baseline : {duration_base:.2f} secondes")
    rm.log(f"Nombre de paramètres du modèle : {params['total']}")
    rm.log(f"Accuracy Baseline sur Amazon (Interne) : {acc_az:.2%}")

    # --- 3. CHARGEMENT & PRÉPARATION TWITTER (DOMAIN B) ---
    rm.log("\n--- Phase 2 : Préparation du Domaine Cible (Twitter) ---")
    df_tw = pd.read_csv(config.PATH_TWITTER)
    df_tw['sentiment'] = df_tw['sentiment'].map(config.TW_MAPPING)
    df_tw = df_tw.dropna(subset=['sentiment'])
    df_tw['text_clean'] = df_tw['content'].apply(clean_text)

    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
        df_tw['text_clean'], df_tw['sentiment'], test_size=0.2, random_state=42, stratify=df_tw['sentiment'])

    # Évaluation Baseline sur Domaine B (Zero-Shot) [Point 16 du Projet]
    acc_zs = np.mean(model_A.predict(X_t_test) == y_t_test)
    rm.log(f"Accuracy Baseline sur Twitter (Zero-Shot) : {acc_zs:.2%}")

    # --- 4. AFFICHAGE DES DISTRIBUTIONS (Point 15 du Projet) ---
    rm.log("\n--- TABLEAU DE RÉPARTITION DES ÉCHANTILLONS ---")
    subsets = {
        "Amazon Train (S_train)": y_train_az,
        "Amazon Test (S_test)": y_test_az,
        "Twitter Pool (T_pool)": y_t_train,
        "Twitter Test (T_test)": y_t_test
    }
    for name, labels in subsets.items():
        rm.log(f"\n{name} - Total: {len(labels)}")
        dist = labels.value_counts(normalize=True) * 100
        rm.log(dist.to_string(float_format="{:.2f}%".format))

    # --- 5. BOUCLE ACTIVE LEARNING ---
    rm.log(f"\n--- Phase 3 : Active Learning Loop ({strategy}) ---")
    results = np.zeros((len(config.BUDGETS), config.N_ITERATIONS))

    for i in range(config.N_ITERATIONS):
        rm.log(f"\n> Itération {i+1}/{config.N_ITERATIONS}")
        current_model = copy.deepcopy(model_A)
        pool_indices = np.arange(len(X_t_train))
        train_indices = []

        for j, ratio in enumerate(config.BUDGETS):
            target_count = int(len(X_t_train) * ratio)
            n_to_add = target_count - len(train_indices)
            
            if n_to_add > 0:
                start_strat = time.time() # Temps de la stratégie
                
                if strategy == 'random':
                    new_indices = np.random.choice(pool_indices, n_to_add, replace=False)
                else:
                    X_pool_text = X_t_train.iloc[pool_indices]
                    probs = current_model.predict_proba(X_pool_text)
                    
                    if strategy == 'entropy':
                        scores = al_strat.get_entropy_scores(probs)
                    elif strategy == 'margin':
                        scores = al_strat.get_margin_scores(probs)
                    elif strategy == 'combined':
                        scores = al_strat.get_combined_scores(probs)
                    elif strategy == 'density':
                        X_pool_vec = current_model.named_steps['tfidf'].transform(X_pool_text)
                        scores = al_strat.get_density_scores(X_pool_vec)
                    elif strategy == 'diversity':
                        X_pool_vec = current_model.named_steps['tfidf'].transform(X_pool_text)
                        local_idx = al_strat.get_k_means_representative_samples(X_pool_vec, n_to_add)
                        new_indices = pool_indices[local_idx]
                    elif strategy == 'max_dist': 
                        X_pool_vec = current_model.named_steps['tfidf'].transform(X_pool_text)
                        X_train_current = X_t_train.iloc[train_indices] if train_indices else X_pool_text[:1]
                        X_train_vec = current_model.named_steps['tfidf'].transform(X_train_current)
                        local_idx = al_strat.get_diverse_max_dist_indices(X_pool_vec, X_train_vec, n_to_add)
                        new_indices = pool_indices[local_idx]
                    
                    if strategy not in ['diversity', 'max_dist']:
                        best_local_indices = np.argsort(scores)[-n_to_add:]
                        new_indices = pool_indices[best_local_indices]

                strat_duration = time.time() - start_strat
                train_indices.extend(new_indices)
                pool_indices = np.array([idx for idx in pool_indices if idx not in new_indices])

                if i == 0 and j == 0:
                    rm.log(f"Temps de sélection ({strategy}) pour le premier palier : {strat_duration:.4f}s")

            # --- ANALYSE QUALITATIVE À 10% (POUR VALIDER LE CRITÈRE À 3 PTS DE LA GRILLE) ---
            if ratio == 0.1 and i == 0:
                rm.log(f"\n--- ANALYSE QUALITATIVE À 10% DE BUDGET ({strategy}) ---")
                text_10 = X_t_train.iloc[train_indices]
                
                avg_len_10 = text_10.str.len().mean()
                excl_10 = text_10.str.count('!').mean()
                
                # Récupération du vocabulaire Amazon pour le calcul du taux d'OOV
                amazon_vocab = model_A.named_steps['tfidf'].vocabulary_
                def count_oov_local(t):
                    words = t.split()
                    if not words: return 0
                    oov = [w for w in words if w not in amazon_vocab]
                    return len(oov) / len(words)
                
                avg_oov_10 = text_10.apply(count_oov_local).mean()
                
                rm.log(f"Longueur moyenne (10%) : {avg_len_10:.2f} car.")
                rm.log(f"Taux d'exclamation (10%) : {excl_10:.2f}")
                rm.log(f"Taux de mots inconnus OOV (10%) : {avg_oov_10:.2%}")
                rm.log("\n--- EXEMPLES DE TWEETS SÉLECTIONNÉS (TOP 5) ---")
                for k, (idx, tweet) in enumerate(text_10.head(5).items()):
                    rm.log(f"Exemple {k+1} (Index {idx}) : {tweet}")
                rm.log("Cette analyse permet de comparer les préférences de sélection à un stade précoce.")

            # Ré-entraînement itératif (Point 29 & 30 du Projet)
            X_train_step = X_t_train.iloc[train_indices]
            y_train_step = y_t_train.iloc[train_indices]
            
            step_model = copy.deepcopy(model_A)
            clf = step_model.named_steps['clf']
            clf.warm_start = True
            
            X_train_vec = step_model.named_steps['tfidf'].transform(X_train_step)
            clf.fit(X_train_vec, y_train_step)
            
            X_test_vec = step_model.named_steps['tfidf'].transform(X_t_test)
            results[j, i] = accuracy_score(y_t_test, clf.predict(X_test_vec))
            current_model = step_model

    # --- 6. ANALYSE FINALE (100% BUDGET) ---
    rm.log(f"\n--- Analyse qualitative des échantillons choisis ({strategy} - 100% Budget) ---")
    last_selected_text = X_t_train.iloc[train_indices]
    
    avg_len = last_selected_text.str.len().mean()
    excl_count = last_selected_text.str.count('!').mean()
    
    amazon_vocab = model_A.named_steps['tfidf'].vocabulary_
    def count_oov_final(text):
        words = text.split()
        if not words: return 0
        oov = [w for w in words if w not in amazon_vocab]
        return len(oov) / len(words)
    
    avg_oov = last_selected_text.apply(count_oov_final).mean()
    
    rm.log(f"Longueur moyenne finale : {avg_len:.2f} car.")
    rm.log(f"Taux d'exclamation final : {excl_count:.2f}")
    rm.log(f"Taux de mots inconnus (OOV) final : {avg_oov:.2%}")
    
    rm.save_learning_curve(config.BUDGETS, results, acc_zs)
    rm.log("\nExpérimentation terminée.")

if __name__ == "__main__":
    main()