import pandas as pd

file_path = "../Data/Dataset_pubmed.csv"
data = pd.read_csv(file_path)

data.head(), data.columns
