"""
Dataset reader
"""
import pandas as pd

def read_csv_file(f_path):
    """
    read the csv file
    """
    return pd.read_csv(f_path)

def peek_dataframe(df, n=5):
    """
    Take a peek into the dataframe
    """
    print(df.shape)
    print(df.columns.tolist())
    return df.head(n)

def read_text_file(f_path):
    """
    Read the text file
    """
    with open(f_path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def read_parquet_files(f_path):
    """
    Read PARQUET files
    """
    return pd.read_parquet(f_path)