"""Module to calculate the scores of the model."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Score(ABC):
    """Class to store model scores."""

    @abstractmethod
    def calculate(self, y_true: Any, y_pred: Any) -> None:
        """Method to calculate the scores."""


@dataclass
class ScoreClassification(Score):
    """Class to store model scores."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    kappa: Optional[float] = None
    confusion_matrix: Optional[Any] = None

    def calculate(self, y_true: Any, y_pred: Any) -> None:
        """Method to calculate the scores."""
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1 = f1_score(y_true, y_pred)
        self.auc_roc = roc_auc_score(y_true, y_pred)
        self.kappa = cohen_kappa_score(y_true, y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)


def root_mean_squared_log_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Logarithmic Error (RMSLE).

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: The RMSLE score.
    """

    return np.sqrt(mean_squared_log_error(y_true, y_pred))


@dataclass
class ScoreRegression(Score):
    """Class to store model scores."""

    r2: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmsle: Optional[float] = None

    def calculate(self, y_true: Any, y_pred: Any) -> None:
        """Method to calculate the scores."""
        self.r2 = r2_score(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)
        self.mae = mean_absolute_error(y_true, y_pred)
        self.rmsle = root_mean_squared_log_error(y_true, y_pred)
