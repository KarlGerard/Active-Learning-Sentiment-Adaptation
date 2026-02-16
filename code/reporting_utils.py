# code/reporting_utils.py

import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class ResultsManager:
    def __init__(self, output_dir, strategy_name):
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.log_path = os.path.join(output_dir, f"rapport_{strategy_name}.txt")
        with open(self.log_path, "w", encoding="utf-8") as f: 
            f.write(f"=== RAPPORT D'EXPERIMENTATION : {strategy_name.upper()} ===\n\n")

    def log(self, text):
        print(text)
        with open(self.log_path, "a", encoding="utf-8") as f: f.write(str(text) + "\n")

    def save_matrix(self, y_true, y_pred, title, filename, cmap='Blues'):
        filepath = os.path.join(self.output_dir, f"{self.strategy_name}_{filename}")
        labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
        plt.title(f"{title}\nAccuracy: {accuracy_score(y_true, y_pred):.2%}")
        plt.savefig(filepath)
        plt.close()
        self.log(f"Matrice enregistrée : {filepath}")
        report = classification_report(y_true, y_pred)
        self.log(str(report) + "\n")

    # Cette méthode doit être indentée au même niveau que save_matrix
    def save_learning_curve(self, budgets, results_matrix, acc_zs):
        mean_scores = np.mean(results_matrix, axis=1)
        std_scores = np.std(results_matrix, axis=1)
        
        # --- CETTE PARTIE EST CELLE QUI MANQUE POUR plot_comparison.py ---
        self.log("\n--- RÉSULTATS DÉTAILLÉS ---")
        self.log("Budget %\tAcc Moyenne %\tEcart-type")
        for b, m, s in zip(budgets, mean_scores, std_scores):
            # On écrit le tableau que plot_comparison va lire
            self.log(f"{b*100:.1f}\t{m*100:.2f}\t{s*100:.2f}")
        # ------------------------------------------------------------------
        
        plt.figure(figsize=(10, 6))
        x_values = [b*100 for b in budgets]
        plt.plot(x_values, mean_scores, marker='o', color='purple', label=f'Stratégie: {self.strategy_name}')
        plt.fill_between(x_values, mean_scores-std_scores, mean_scores+std_scores, alpha=0.15, color='purple')
        plt.axhline(y=acc_zs, color='red', linestyle='--', label=f'Zero-Shot ({acc_zs:.3f})')
        plt.title(f"Courbe d'apprentissage - {self.strategy_name}")
        plt.xlabel("% du Dataset Twitter")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"courbe_{self.strategy_name}.png"))
        plt.close()