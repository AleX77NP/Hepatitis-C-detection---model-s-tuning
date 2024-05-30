import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(file: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(file)
    return df

def split_data(X, label, test_size=0.25, random_state=2):
    y=X[label]
    unique_classes = np.unique(y)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for class_value in unique_classes:
        X_class = X[y == class_value]
        y_class = y[y == class_value]
        X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=test_size, random_state=random_state)
        X_train.append(X_class_train)
        X_test.append(X_class_test)
        y_train.append(y_class_train)
        y_test.append(y_class_test)
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    return X_train, X_test, y_train, y_test