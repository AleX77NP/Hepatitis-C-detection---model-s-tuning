import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from autoviz.AutoViz_Class import AutoViz_Class

dataset_path = 'dataset/hcvdat0.csv'


def load_dataset(file: str) -> pd.DataFrame:
    df: DataFrame = pd.read_csv(file)
    return df


hcv_df = load_dataset(dataset_path)
hcv_df = hcv_df[hcv_df.columns[1:]]  # remove first 'Unnamed' column
# print(hcv_df.head(5))

print(f'Number of different categories: {hcv_df["Category"].value_counts()}')
# describe DataFrame
print(hcv_df.describe())

# creating an AutoViz instance
AV = AutoViz_Class()

# generating data visualization automatically
AV.AutoViz(
    filename='',
    sep=',',
    depVar='',
    dfte=hcv_df,
    header=0,
    verbose=0,
    lowess=False,
    chart_format='svg',
    max_rows_analyzed=10000,
    max_cols_analyzed=30
)
