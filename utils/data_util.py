import os
import pandas as pd


def text_to_dataframe(file_path: str, columns: list) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = [line.strip().split('\t') for line in lines]
    dataframe = pd.DataFrame(data, columns=columns)

    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column])

    return dataframe
