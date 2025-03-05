import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Charger les données depuis un fichier .parquet
file_path = "new_data.parquet"  # Remplacez par le chemin de votre fichier

df = pd.read_parquet(file_path)

# Convertir le timestamp en date lisible
df['ts'] = pd.to_datetime(df['ts'], unit='s')

# Nettoyer la colonne 'value' (supprimer les guillemets)
df['value'] = df['value'].astype(str).str.replace('"', '', regex=True).astype(int)

# Définir une palette de couleurs
unique_types = df['contentType'].unique()
palette = dict(zip(unique_types, sns.color_palette(n_colors=len(unique_types))))

# Tracer les données
plt.figure(figsize=(12, 6))
for contentType in unique_types:
    subset = df[df['contentType'] == contentType]
    plt.plot(subset['ts'], subset['value'], marker='o', linestyle='-', label=contentType, color=palette[contentType])

plt.xlabel("Date")
plt.ylabel("Valeur")
plt.title("Visualisation des données par ContentType")
plt.legend()
# plt.xticks(rotation=45)
plt.grid(True)
plt.show()
