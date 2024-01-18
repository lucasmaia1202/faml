"""Module to help with exploratory data analysis."""

from typing import Callable, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def handle_numeric_missing_values(
    data: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    This function handles numeric missing values in the dataset.

    Atributes:
        data: pd.DataFrame - The dataset to be handled.
        columns: List[str] - The columns to be handled.

    Returns:
        data: pd.DataFrame - The dataset with the missing values handled.
    """

    for column in columns:
        data[column] = data[column].fillna(data[column].median())

    return data


def handle_categorical_missing_values(
    data: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    This function handles categorial missing values in the dataset.

    Atributes:
        data: pd.DataFrame - The dataset to be handled.
        columns: List[str] - The columns to be handled.

    Returns:
        data: pd.DataFrame - The dataset with the missing values handled.
    """

    for column in columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    return data


def handle_one_hot_encoding(
    data: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    This function handles categorial one hot encoding in the dataset.

    Atributes:
        data: pd.DataFrame - The dataset to be handled.
        columns: List[str] - The columns to be handled.

    Returns:
        data: pd.DataFrame - The dataset with the new columns handled.
    """

    # create instance of one-hot encoder
    encoder = OneHotEncoder(handle_unknown="ignore")
    new_columns = []
    for col in columns:
        print("Encoding column: ", col)
        encoder.fit(data[[col]])
        encoded_col = encoder.transform(data[[col]])
        encoded_df = pd.DataFrame(encoded_col.toarray())
        encoded_df.columns = [
            f"{col}_{category}" for category in encoder.categories_[0]
        ]
        new_columns.extend(encoded_df.columns)
        data.drop(col, axis=1, inplace=True)
        data = pd.concat([data, encoded_df], axis=1)

    return data, new_columns


def handle_scaling(
    data: pd.DataFrame,
    columns: List[str],
    scaler: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    This function handles scaling in the dataset.

    Atributes:
        data: pd.DataFrame - The dataset to be handled.
        columns: List[str] - The columns to be handled.

    Returns:
        data: pd.DataFrame - The dataset with the new columns handled.
    """

    if scaler is None:
        scaler = RobustScaler()

    for col in columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    return data
