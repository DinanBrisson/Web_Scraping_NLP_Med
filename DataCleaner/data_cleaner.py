import pandas as pd

# Chemins des fichiers
pubmed_data_path = '../Data/Dataset_pubmed.csv'
scimago_data_path = '../Data/scimagojr 2023.csv'

# Charger les fichiers
pubmed_data = pd.read_csv(pubmed_data_path)
scimago_data = pd.read_csv(scimago_data_path, delimiter=';')

# Préparer les données de Scimago : conserver seulement les colonnes nécessaires
scimago_data = scimago_data[['Title', 'H index']].dropna(subset=['H index'])

# Convertir 'H index' en numérique
scimago_data['H index'] = pd.to_numeric(scimago_data['H index'], errors='coerce')

# Fusionner les données PubMed avec les H-index des journaux depuis Scimago
pubmed_data = pubmed_data.merge(scimago_data[['Title', 'H index']],
                                left_on='Journal', right_on='Title', how='left')

# Sauvegarder le fichier enrichi avec la colonne H-index
output_path = 'Dataset_pubmed_with_H_index.csv'  # Chemin de sortie
pubmed_data.to_csv(output_path, index=False)

print(f"Le fichier enrichi avec les H-index a été sauvegardé sous : {output_path}")
