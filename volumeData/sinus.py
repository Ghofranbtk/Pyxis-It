import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Charger les données depuis le fichier txt
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Ignorer l'en-tête
    hours = data[:, 0]
    values = data[:, 1]
    return hours, values

# Détection des pics maximum et minimum
def detect_extrema(x, y):
    peaks_max, _ = find_peaks(y, distance=5)  # Pics maximums
    peaks_min, _ = find_peaks(-y, distance=5)  # Pics minimums
    return x[peaks_max], y[peaks_max], x[peaks_min], y[peaks_min]

# Définition d'une fonction gaussienne
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

# Définition d'une fonction sinusoïdale
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Ajustement des données avec gestion d'erreur
def fit_functions(x, y):
    try:
        p0_gauss = [max(y), np.mean(x), np.std(x) or 1]  # Éviter std(x) = 0
        popt_gauss, _ = curve_fit(gaussian, x, y, p0=p0_gauss, maxfev=5000)
    except RuntimeError:
        print("⚠️ Ajustement Gaussien impossible.")
        popt_gauss = [0, 0, 1]  # Valeurs par défaut
    
    try:
        p0_sinus = [max(y) - min(y), 2 * np.pi / max((x[-1] - x[0]), 1), 0, np.mean(y)]
        popt_sinus, _ = curve_fit(sinusoidal, x, y, p0=p0_sinus, maxfev=5000)
    except RuntimeError:
        print("⚠️ Ajustement Sinusoïdal impossible.")
        popt_sinus = [0, 1, 0, np.mean(y)]

    return popt_gauss, popt_sinus

# Affichage des résultats
def plot_results(x, y, peaks_max_x, peaks_max_y, peaks_min_x, peaks_min_y):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=5, label="Données", color='black')
    plt.scatter(peaks_max_x, peaks_max_y, color='red', label="Pics maximums")
    plt.scatter(peaks_min_x, peaks_min_y, color='blue', label="Pics minimums")
    
    # Tracer les ajustements si disponibles
    

    plt.legend()
    plt.xlabel("Temps (heures)")
    plt.ylabel("Valeurs")
    plt.title("Analyse des données - Ajustements Gaussien et Sinusoïdal")
    plt.show()

# Main
filename = "points_reduits_by_hour.txt"  # Remplace par le nom réel de ton fichier
hours, values = load_data(filename)
peaks_max_x, peaks_max_y, peaks_min_x, peaks_min_y = detect_extrema(hours, values)
# popt_gauss, popt_sinus = fit_functions(hours, values)

# Afficher les résultats
plot_results(hours, values, peaks_max_x, peaks_max_y, peaks_min_x, peaks_min_y)
