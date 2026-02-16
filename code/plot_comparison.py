import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import config

def extract_scores(strategy: str):
    """Extrait les scores ET l'écart-type du fichier rapport."""
    path = os.path.join(config.OUTPUT_DIR, f"rapport_{strategy}.txt")
    
    if not os.path.exists(path): 
        return [], [], []
    
    budgets, scores, stds = [], [], []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        start_reading = False
        for line in lines:
            if "Budget %" in line:
                start_reading = True
                continue
            
            if start_reading and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        b = float(parts[0])
                        s = float(parts[1]) / 100.0
                        # On extrait l'écart-type si la 3ème colonne existe
                        st = float(parts[2]) / 100.0 if len(parts) >= 3 else 0.0
                        
                        budgets.append(b)
                        scores.append(s)
                        stds.append(st)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Erreur lors de la lecture de {strategy}: {e}")
        return [], [], []

    return budgets, scores, stds

def plot_all():
    """Génère le graphique final avec zone d'ombre pour l'incertitude."""
    plt.figure(figsize=(12, 8))
    
    strategies = ['random', 'entropy', 'margin', 'diversity', 'density', 'max_dist', 'combined']
    colors = {
        'random': 'gray', 'entropy': 'blue', 'margin': 'cyan', 
        'diversity': 'green', 'density': 'olive', 'max_dist': 'lime', 'combined': 'purple'
    }
    
    has_data = False
    
    for strat in strategies:
        b, s, st = extract_scores(strat)
        
        if len(b) > 0:
            # On trace la ligne principale
            color = colors.get(strat, 'black')
            plt.plot(b, s, marker='o', label=f"Stratégie: {strat}", color=color, linewidth=2)
            
            # AJOUT : Si l'écart-type est présent (cas du random), on trace la zone grisée
            if any(val > 0 for val in st):
                s_arr = np.array(s)
                st_arr = np.array(st)
                plt.fill_between(b, s_arr - st_arr, s_arr + st_arr, color=color, alpha=0.15)
            
            has_data = True
            
    if not has_data:
        print("Aucune donnée trouvée dans /results.")
        return

    plt.title("Comparaison des performances d'Active Learning (Domaine Cible)")
    plt.xlabel("% du Dataset Twitter utilisé (Budget)")
    plt.ylabel("Précision (Accuracy)")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(config.OUTPUT_DIR, "comparaison_finale.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique final généré avec zone de confiance : {output_path}")

if __name__ == "__main__":
    plot_all()