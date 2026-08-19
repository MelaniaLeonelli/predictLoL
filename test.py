import pandas as pd

from prep_functions import (
    rimuovi_collinearita,
    rimuovi_ridondanze_speculari,
    rimuovi_aggregati
)

df = pd.read_csv("data/high_diamond_ranked_10min.csv")

print("Colonne iniziali:", df.shape[1])

df_test = rimuovi_collinearita(df)
df_test = rimuovi_ridondanze_speculari(df_test)
df_test = rimuovi_aggregati(df_test)

print("Colonne finali:", df_test.shape[1])
print(df_test.columns.tolist())