import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Charger les données depuis un fichier .parquet
file_path = "new_data.parquet"  # Remplacez par le chemin de votre fichier
df = pd.read_parquet(file_path)

# Filtrer les données pour contentType = FTP
df_ftp = df.loc[df['contentType'] == 'FTP'].copy()

# Convertir le timestamp en date lisible
df_ftp.loc[:, 'ts'] = pd.to_datetime(df_ftp['ts'], unit='s')

# Trier les données par timestamp croissant
df_ftp = df_ftp.sort_values(by='ts')

# Sauvegarder les données filtrées et triées dans un fichier CSV
df_ftp.to_csv("ftp_data.csv", index=False)

# Calculer la différence entre les timestamps
df_ftp.loc[:, 'ts_diff'] = df_ftp['ts'].diff().dt.total_seconds()

time_diff = df_ftp['ts_diff'].dropna().mean()
time_unit = "heures" if time_diff < 86400 else "jours"

# Tracer les données
plt.figure(figsize=(12, 6))
plt.plot(df_ftp['ts'], df_ftp['value'].astype(int), marker='o', linestyle='-', label="FTP", color='blue')
plt.xlabel("Date")
plt.ylabel("Valeur")
plt.title(f"Évolution des valeurs FTP ({time_unit})")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
