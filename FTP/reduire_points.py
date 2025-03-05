import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def lire_points(fichier):
    """ Lit un fichier CSV contenant les colonnes contentType, ts, value et retourne un tableau NumPy """
    df = pd.read_csv(fichier, usecols=["ts", "value"], parse_dates=["ts"])  # Lire les colonnes nécessaires
    df["ts"] = df["ts"].astype("int64") // 10**9  # Convertir timestamp en secondes
    df["value"] = df["value"].astype(float)  # S'assurer que 'value' est bien numérique
    return df.values  # Convertir en tableau NumPy


def calculer_pente(point1, point2):
    """ Calcule la pente entre deux points (x1, y1) et (x2, y2). """
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

def filtrer_points_par_plage_et_colinearite(points, epsilon, seuil_pente):
    """ Conserve seulement les points aux extrémités des plages successives proches 
        et élimine les points intermédiaires collinéaires.
    """
    reduced_points = [points[0]]  # Commence avec le premier point
    
    for i in range(1, len(points) - 1):
        # Si la différence en Y est inférieure à epsilon, c'est une plage
        if np.abs(points[i][1] - points[i - 1][1]) < epsilon and np.abs(points[i + 1][1] - points[i][1]) < epsilon:
            continue  # Ignorer les points intermédiaires dans la plage
        
        # Vérification de la colinéarité des trois points successifs
        pente1 = calculer_pente(points[i - 1], points[i])
        pente2 = calculer_pente(points[i], points[i + 1])
        
        if np.abs(pente1 - pente2) < seuil_pente:  # Si les pentes sont proches, points collinéaires
            continue  # Ignorer le point intermédiaire
        
        # Ajouter le point actuel
        reduced_points.append(points[i])
    
    # Ajouter le dernier point
    reduced_points.append(points[-1])
    
    return np.array(reduced_points)

def calculer_taux_erreur(points_originaux, points_reduits):
    """ Calcule le taux d'erreur entre les points d'origine et les points réduits en interpolant les valeurs manquantes. """
    # Interpolation linéaire sur les points réduits
    interpolation = interp1d(points_reduits[:, 0], points_reduits[:, 1], kind='linear', fill_value="extrapolate")
    
    # Estimation des Y pour les X des points d'origine
    y_interpoles = interpolation(points_originaux[:, 0])
    
    # Calcul du taux d'erreur basé sur la différence absolue normalisée
    erreur_absolue = np.abs(points_originaux[:, 1] - y_interpoles)
    erreur_totale = np.sum(erreur_absolue)
    somme_y_original = np.sum(points_originaux[:, 1])
    
    taux_erreur = (erreur_totale / somme_y_original) * 100  # En pourcentage
    return taux_erreur

# Nom des fichiers
fichier_points = "ftp_data.csv"  # Fichier CSV en entrée
fichier_reduit = "points_reduits.txt"  # Fichier CSV en sortie

# Lecture des points
points = lire_points(fichier_points)

# Application de la réduction avec un seuil donné
epsilon = 1e2  # Seuil pour la différence en Y
seuil_pente = 1e2  # Seuil pour la différence de pente
reduced_points = filtrer_points_par_plage_et_colinearite(points, epsilon, seuil_pente)

# Enregistrement des points réduits sous format CSV
pd.DataFrame(reduced_points, columns=['X', 'Y']).to_csv(fichier_reduit, index=False)

# Calcul du taux d'erreur
taux_erreur = calculer_taux_erreur(points, reduced_points)
print(f"Taux d'erreur (perte d'information) : {taux_erreur:.2f}%")

# Visualisation
plt.figure(figsize=(10, 5))
plt.scatter(points[:, 0], points[:, 1], color='r', label="Points d'origine")
plt.plot(points[:, 0], points[:, 1], 'b-', alpha=0.5)  # Relie les points d'origine
plt.scatter(reduced_points[:, 0], reduced_points[:, 1], color='g', label="Points après réduction", zorder=3)
plt.plot(reduced_points[:, 0], reduced_points[:, 1], 'g-', alpha=0.5)  # Relie les points réduits
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Réduction des points (Taux d'erreur : {taux_erreur:.2f}%)")
plt.grid(True)
plt.show()
