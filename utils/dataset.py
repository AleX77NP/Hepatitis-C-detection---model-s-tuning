import pandas as pd

def load_dataset(file: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(file)
    return df