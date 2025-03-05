import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier CSV
data = pd.read_csv('jour.csv')

# Convertir la colonne 'Date' en datetime pour une gestion correcte
data['Date'] = pd.to_datetime(data['Date'])

# Tracer les points et relier les points avec une ligne
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['value'], color='blue', marker='o', label='Valeur par Date')  # Relier les points
plt.scatter(data['Date'], data['value'], color='red')  # Afficher les points en rouge

# Ajouter des étiquettes et un titre
plt.title('Valeur par Date')
plt.xlabel('Date')
plt.ylabel('Valeur')

# Ajouter une grille pour améliorer la lisibilité
plt.grid(True)

# Afficher le graphique
plt.xticks(rotation=45)  # Rotation des dates pour plus de lisibilité
plt.tight_layout()
plt.show()
