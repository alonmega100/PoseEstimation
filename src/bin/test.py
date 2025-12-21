import pandas as pd

from src.utils.tools import choose_csv_interactively


csv_file = choose_csv_interactively("data/CSV")

df = pd.read_csv(csv_file)

cam1 = df[df["source"] == "839112060834"]
cam2 = df[df["source"] == "839112062097"]


for i in range(3):
    col = f"pose_{i}3"

    mask1 = df["source"] == "839112060834"
    df.loc[mask1, col] -= df.loc[mask1, col].mean()

    mask2 = df["source"] == "839112062097"
    df.loc[mask2, col] -= df.loc[mask2, col].mean()

df.to_csv(csv_file, index=False)
