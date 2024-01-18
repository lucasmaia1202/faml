"""Class to store machine learning model information."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from utils.score import ScoreClassification


class SeachMethod(Enum):
    BAYES = "bayes"
    RANDOM = "random"
    GRID = "grid"


@dataclass
class ModelProtocol(Protocol):
    """Protocol to be followed by all models."""

    def fit(self, x: Any, y: Any) -> Any:
        """Method to train the model."""

    def predict(self, x: Any) -> Any:
        """Method to predict with the model."""


@dataclass
class ModelOptimizer:
    """Class to store machine learning model information."""

    name: str
    model: ModelProtocol
    search_space: Optional[Dict[str, Any]] = None
    best: Optional[ModelProtocol] = None
    scores: Optional[ScoreClassification] = None
    search_method: SeachMethod = SeachMethod.BAYES
    k_fold: int = 3

    def __hyper_tune(self, x: Any, y: Any) -> None:
        print("Hyper tuning:", self.search_method)
        if self.search_method == SeachMethod.BAYES:
            opt = BayesSearchCV(
                self.model,
                self.search_space,
                n_iter=50,
                cv=self.k_fold,
                scoring=make_scorer(roc_auc_score),
            )
        elif self.search_method == SeachMethod.RANDOM:
            opt = RandomizedSearchCV(
                self.model,
                self.search_space,
                n_iter=50,
                cv=self.k_fold,
                scoring=make_scorer(roc_auc_score),
            )
        elif self.search_method == SeachMethod.GRID:
            opt = GridSearchCV(
                self.model,
                self.search_space,
                cv=self.k_fold,
                scoring=make_scorer(roc_auc_score),
                verbose=3,
            )
        else:
            raise ValueError("Invalid search method.")

        opt.fit(x, y)
        self.best = opt.best_estimator_
        print("Best params:", opt.best_params_)

    def train(self, x: Any, y: Any) -> None:
        """Method to train the model."""
        print("Training model:", self.name)
        if self.search_space is None:
            self.model.fit(x, y)
            self.best = self.model
        else:
            self.__hyper_tune(x, y)

    def evaluate(self, x: Any, y: Any) -> None:
        """Method to evaluate the model."""
        y_pred = self.best.predict(x)
        self.scores = ScoreClassification()
        self.scores.calculate(y, y_pred)
