import pandas as pd

# Charger les données depuis le fichier CSV
data = pd.read_csv('ftp_data.csv', header=None, names=['Type', 'Timestamp', 'Value'])

# Convertir la colonne 'Value' en type numérique, en forçant les erreurs à devenir NaN
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')

# Filtrer les lignes où 'Timestamp' est une date valide
data = data[pd.to_datetime(data['Timestamp'], errors='coerce').notnull()]

# Convertir la colonne 'Timestamp' en type datetime (en ignorant les erreurs)
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Calculer les différences entre les valeurs consécutives
data['Value_diff'] = data['Value'].diff().abs()

# Calculer le seuil comme un pourcentage de la différence maximale
max_diff = data['Value_diff'].max()
threshold = max_diff * 0.1  # Par exemple, 10% de la différence maximale

print(f"Seuil pour ignorer les points similaires : {threshold}")

# Filtrer les valeurs où la différence est inférieure au seuil
ignored_points = data[data['Value_diff'] < threshold]

# Afficher les points qui peuvent être ignorés
print("Points à ignorer (différence inférieure au seuil) :")
print(ignored_points)
