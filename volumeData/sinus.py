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

# Détection des pics maximum et minimum avec une meilleure gestion du premier point
def detect_extrema(x, y):
    peaks_max, _ = find_peaks(y, distance=3, prominence=0)  # Pics maximums avec une meilleure détection
    peaks_min, _ = find_peaks(-y, distance=3, prominence=0)  # Pics minimums avec une meilleure détection
    
    # Ajouter la gestion du premier point (vérifier si c'est un pic local)
    if y[0] > y[1]:
        peaks_max = np.insert(peaks_max, 0, 0)
    elif y[0] < y[1]:
        peaks_min = np.insert(peaks_min, 0, 0)
        
    return peaks_max, peaks_min

# Détection des pics locaux maximums entre chaque paire de minima
def detect_local_max_between_minima(x, y, minima, threshold=0.1):
    local_peaks_max = []
    
    # Parcourir chaque intervalle entre deux minima consécutifs
    for i in range(len(minima) - 1):
        start_idx = minima[i]
        end_idx = minima[i + 1]
        
        # Segment entre les deux minima consécutifs
        segment_x = x[start_idx:end_idx]
        segment_y = y[start_idx:end_idx]
        
        # Trouver les pics dans ce segment
        peaks_max, _ = find_peaks(segment_y, distance=3, prominence=0)
        
        # Vérifier qu'il existe un pic local dans cet intervalle
        if len(peaks_max) > 0:
            # On sélectionne le pic le plus élevé dans cet intervalle
            max_peak_idx = peaks_max[np.argmax(segment_y[peaks_max])]
            if segment_y[max_peak_idx] - min(segment_y) > threshold:
                local_peaks_max.append(max_peak_idx + start_idx)  # Ajuster l'indice pour correspondre aux données globales
            else:
                # S'il n'y a pas un pic significatif, vérifier si une condition alternative doit être appliquée
                local_peaks_max.append(np.argmax(segment_y) + start_idx)
                
    return np.array(local_peaks_max)

# Définition d'une fonction gaussienne
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

# Définition d'une fonction sinusoïdale
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Ajustement des segments de données
def fit_segments(x, y, peaks):
    segment_params = []
    for i in range(len(peaks) - 1):
        x_segment = x[peaks[i]:peaks[i+1]]
        y_segment = y[peaks[i]:peaks[i+1]]
        
        # Skip segments that contain fewer than 4 points (for sinusoidal fitting)
        if len(x_segment) < 4:
            continue  # Ignore this segment
        
        try:
            popt_gauss, _ = curve_fit(gaussian, x_segment, y_segment, 
                                      p0=[max(y_segment), np.mean(x_segment), np.std(x_segment)], 
                                      maxfev=5000)
        except RuntimeError:
            popt_gauss = [0, 0, 1]
        
        try:
            # Only fit sinusoidal if there are at least 4 data points
            popt_sinus, _ = curve_fit(sinusoidal, x_segment, y_segment, 
                                      p0=[max(y_segment) - min(y_segment), 
                                          2 * np.pi / max((x_segment[-1] - x_segment[0]), 1), 
                                          0, np.mean(y_segment)], 
                                      maxfev=5000)
        except RuntimeError:
            popt_sinus = [0, 1, 0, np.mean(y_segment)]
        
        segment_params.append((popt_gauss, popt_sinus, x_segment))
    
    return segment_params

# Affichage des résultats
def plot_results(x, y, peaks_max, peaks_min, local_peaks_max, segment_params):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=5, label="Données", color='black')
    plt.scatter(x[peaks_max], y[peaks_max], color='red', label="Pics maximums")
    plt.scatter(x[peaks_min], y[peaks_min], color='blue', label="Pics minimums")
    
    # Affichage des pics locaux maximums avant chaque minimum en violet
    plt.scatter(x[local_peaks_max], y[local_peaks_max], color='violet', label="Pics locaux max entre minima")
    
    plt.legend()
    plt.xlabel("Temps (heures)")
    plt.ylabel("Valeurs")
    plt.title("Analyse des données - Pics locaux et extrêmes")
    plt.show()

# Main
filename = "points_reduits_by_hour.txt"  # Remplace par le nom réel de ton fichier
hours, values = load_data(filename)
peaks_max, peaks_min = detect_extrema(hours, values)

# Détecter les pics locaux maximums entre chaque paire de minima
local_peaks_max = detect_local_max_between_minima(hours, values, peaks_min, threshold=0.1)

# Ajuster les segments de données (si nécessaire)
segment_params = fit_segments(hours, values, sorted(np.concatenate((peaks_max, peaks_min))))

# Afficher les résultats
plot_results(hours, values, peaks_max, peaks_min, local_peaks_max, segment_params)
