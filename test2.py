import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import ruptures as rpt
import warnings

# === 1. LECTURE ET NORMALISATION DES DONNÉES ===
def lire_donnees(fichier):
    df = pd.read_csv(fichier)
    df['X'] = (df['X'] - df['X'].min()) / 86400  # Convertir X en jours
    df['Y'] = (df['Y'] - df['Y'].min()) / (df['Y'].max() - df['Y'].min())  # Normalisation Y
    print(df)
    return df.dropna()

# === 2. MODÈLES MATHÉMATIQUES ===
def modele_lineaire(x, a, b):
    return a * x + b

def modele_gaussien(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

def modele_sinusoide(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def modele_exponentiel(x, a, b, c):
    return a * np.exp(np.clip(b * x, -100, 100)) + c  

# === 3. FONCTION D'AJUSTEMENT DES MODÈLES ===
from scipy.optimize import OptimizeWarning
import warnings

def ajuster_modele(x, y, modele, p0=None, bounds=(-np.inf, np.inf)):
    if len(x) < 2:  # Trop peu de points
        return float('inf'), None, None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            params, _ = curve_fit(modele, x, y, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = modele(x, *params)
        erreur = mean_squared_error(y, y_pred)
        return erreur, params, y_pred
    except Exception as e:
        print(f"Erreur lors de l'ajustement du modèle : {e}")
        return float('inf'), None, None

# === 4. SEGMENTATION AUTOMATIQUE AVEC PARAMÈTRE AJUSTÉ ===
def segmenter_donnees(x, y):
    signal = np.array(y).reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    changements = algo.predict(pen=0)  # Réduire `pen` pour avoir plus de segments
    segments = []
    print('changements',changements)
    changements = [3, 4, 8, 11, 15, 18,21]  # Vous pouvez ajuster ces points dynamiquement ou les rendre plus intelligents

    debut = 0
    for fin in changements:
        if fin - debut > 1:  # Éviter les segments trop petits
            segments.append((np.array(x[debut:fin]), np.array(y[debut:fin])))
        debut = fin

    return segments

# === 5. ANALYSE ET CHOIX DU MEILLEUR MODÈLE PAR SEGMENT ===
def analyser_segments(fichier):
    data = lire_donnees(fichier)
    x_data, y_data = data['X'].values, data['Y'].values
    segments = segmenter_donnees(x_data, y_data)

    plt.figure(figsize=(12, 6))

    # Enlever cette ligne pour afficher tous les segments
    # segments = segments[:2]  # Cette ligne est maintenant supprimée pour afficher tous les segments

    for i, (x_seg, y_seg) in enumerate(segments):
        erreurs = {}

        erreurs['Linéaire'], params_lin, y_pred_lin = ajuster_modele(x_seg, y_seg, modele_lineaire)
        erreurs['Gaussien'], params_gauss, y_pred_gauss = ajuster_modele(
            x_seg, y_seg, modele_gaussien, 
            p0=[1, np.mean(x_seg), 1], 
            bounds=([0, min(x_seg), 0], [np.inf, max(x_seg), np.inf])  
        )

        erreurs['Sinusoïdal'], params_sin, y_pred_sin = ajuster_modele(
            x_seg, y_seg, modele_sinusoide, 
            p0=[0.5, 2*np.pi, 0, np.mean(y_seg)],
            bounds=([0, 0, -np.pi, -1], [np.inf, 10*np.pi, np.pi, 1])  
        )

        erreurs['Exponentiel'], params_exp, y_pred_exp = ajuster_modele(
            x_seg, y_seg, modele_exponentiel, 
            p0=[1, -0.1, np.min(y_seg)], 
            bounds=([0, -5, -1], [np.inf, 0.5, 1])  
        )

        # Afficher tous les modèles
        for modele, erreur in erreurs.items():
            print(f"Segment {i+1} - Erreur {modele}: {erreur}")

        erreurs = {k: v for k, v in erreurs.items() if v != float('inf')}
        meilleur_modele = min(erreurs, key=erreurs.get) if erreurs else "Aucun"

        print(f"Segment {i+1} - Meilleur modèle : {meilleur_modele}")

        # Afficher les coefficients pour chaque modèle
        if params_lin is not None:
            print(f"  Coefficients du modèle Linéaire (a, b) : {params_lin}")
        if params_gauss is not None:
            print(f"  Coefficients du modèle Gaussien (a, b, c) : {params_gauss}")
        if params_sin is not None:
            print(f"  Coefficients du modèle Sinusoïdal (a, b, c, d) : {params_sin}")
        if params_exp is not None:
            print(f"  Coefficients du modèle Exponentiel (a, b, c) : {params_exp}")

        # Si le modèle choisi est gaussien, attribuer amplitude et sigma
        if meilleur_modele == 'Gaussien' and params_gauss is not None:
            amplitude = params_gauss[0]  # Valeur de y_max
            sigma = params_gauss[2]  # Valeur de x correspondant à y_max
            y_last = y_seg[-1]  # Dernière valeur de y dans le segment
            sigma = y_last
            print(f"  Amplitude (y_max) : {amplitude}")
            print(f"  Sigma (x correspondant à y_max) : {sigma}")

        # Afficher le meilleur modèle pour le segment
        plt.scatter(x_seg, y_seg, label=f"Segment {i+1} - Données")

        if meilleur_modele == 'Linéaire' and params_lin is not None:
            plt.plot(x_seg, y_pred_lin, label="Linéaire", linestyle="dashed")
        elif meilleur_modele == 'Gaussien' and params_gauss is not None:
            plt.plot(x_seg, y_pred_gauss, label="Gaussien", linestyle="dashed")
        elif meilleur_modele == 'Sinusoïdal' and params_sin is not None:
            plt.plot(x_seg, y_pred_sin, label="Sinusoïdal", linestyle="dashed")
        elif meilleur_modele == 'Exponentiel' and params_exp is not None:
            plt.plot(x_seg, y_pred_exp, label="Exponentiel", linestyle="dashed")

    plt.xlabel("Temps (jours)")
    plt.ylabel("Y (normalisé)")
    plt.title("Segmentation et Ajustement des Modèles")
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple de fichier de données
fichier_data = "points_reduits.txt"
analyser_segments(fichier_data)
