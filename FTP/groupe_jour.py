import pandas as pd

# Lire le fichier CSV
data = pd.read_csv("ftp_data.csv", header=None, names=["contentType", "ts", "value"])

# Convertir la colonne 'ts' en datetime
data['ts'] = pd.to_datetime(data['ts'], errors='coerce')

# Convertir la colonne 'value' en numérique (en cas de problème de type)
data['value'] = pd.to_numeric(data['value'], errors='coerce')

# Regrouper par jour (en extrayant la date sans l'heure) et calculer la somme des valeurs
data['Date'] = data['ts'].dt.date
grouped_data = data.groupby(['contentType', 'Date'], as_index=False)['value'].sum()

# Sauvegarder dans un fichier CSV
grouped_data.to_csv("jour.csv", index=False)

# Afficher les résultats pour vérifier
print(grouped_data)
