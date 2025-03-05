import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    """ Fonction gaussienne """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Charger les données depuis un fichier CSV
file_path = "points_reduits.csv"  # Remplacez par votre fichier CSV
data = pd.read_csv(file_path)
timestamps = data['Timestamp'].values
values = data['Value'].values

def process_subset(x_data, y_data):
    """ Ajuste une fonction gaussienne sur un sous-ensemble de points. """
    A = max(y_data)
    mu = np.mean(x_data)  
    sigma = np.std(x_data)  # Utilisation de l'écart-type réel
    
    p0 = [A, mu, sigma]
    
    try:
        params, _ = curve_fit(gaussian, x_data, y_data, p0=p0)
        return params
    except RuntimeError:
        print(f"⚠️ Échec de l'ajustement pour x ∈ [{x_data[0]}, {x_data[-1]}]")
        return None

# Fonction pour tester et ajuster la fonction gaussienne sur des sous-ensembles d'une taille variable
def fit_gaussian_for_subsets(x_data, y_data, min_points=5, max_points=20, step=5):
    """ Essaie d'ajuster une gaussienne sur des sous-ensembles de taille variable. """
    num_subsets = len(x_data) // step
    colors = plt.cm.viridis(np.linspace(0, 1, num_subsets))  # Choisir une palette de couleurs

    plt.figure(figsize=(10, 6))

    for idx, start_idx in enumerate(range(0, len(x_data) - min_points + 1, step)):
        # Taille du sous-ensemble entre min_points et max_points
        end_idx = min(start_idx + max_points, len(x_data))
        
        x_subset = x_data[start_idx:end_idx]
        y_subset = y_data[start_idx:end_idx]
        
        params = process_subset(x_subset, y_subset)
        if params is not None:  
            A_fit, mu_fit, sigma_fit = params
            x_dense = np.linspace(x_subset[0], x_subset[-1], 200)  # Plus de points pour un affichage plus lisse
            y_fit = gaussian(x_dense, *params)
            
            # Tracer les résultats sur la même figure
            plt.plot(x_subset, y_subset, 'o', color=colors[idx], label=f'Sous-ensemble {idx + 1}')
            plt.plot(x_dense, y_fit, linestyle='--', color=colors[idx], label=f'Gaussienne {idx + 1}')
    
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Ajustement gaussien sur différents sous-ensembles")
    plt.show()

# Appliquer l'ajustement gaussien à partir des données du fichier CSV
fit_gaussian_for_subsets(timestamps, values, min_points=5, max_points=15, step=5)
