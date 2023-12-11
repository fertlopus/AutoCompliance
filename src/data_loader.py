import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Iterator, Tuple


def load_data_in_chunks(file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """
    Load the dataset in chunks to handle large files efficiently.

    Args:
    file_path (str): The path to the dataset file.
    chunksize (int): Number of rows per chunk.

    Returns:
    Iterator[pd.DataFrame]: An iterator over chunks of the dataset.
    """
    try:
        return pd.read_csv(file_path, chunksize = chunk_size)
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")


def preprocess_data(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a chunk of the dataset (e.g., cleaning, feature extraction).

    Args:
    chunk (pd.DataFrame): A chunk of the dataset to be preprocessed.

    Returns:
    pd.DataFrame: The preprocessed chunk of the dataset.
    """
    try:
        # Here any additional preprocessing steps can be added for any specific requirements
        chunk = chunk.dropna(subset = ["narrative", "product"]).reset_index(drop = True)
        return chunk[["product", "narrative"]]
    except Exception as e:
        raise RuntimeError(f"Error preprocessing data chunk {chunk}: {e}")


def split_data(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.

    Args:
    data (pd.DataFrame): The dataset to split.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and testing datasets.
    """
    try:
        return train_test_split(data, test_size = test_size, random_state = 42)
    except Exception as e:
        raise RuntimeError(f"Error splitting data: {e}")
