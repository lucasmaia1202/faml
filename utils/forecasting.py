"""Forecasting utilities."""
import pandas as pd


class BoostedHybrid:
    """Class to combine two models."""

    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None
        self.y_fit = None
        self.y_resid = None

    def fit(self, x, y):
        """Fit the model to the data."""
        self.model_1.fit(x, y)
        y_fit = pd.DataFrame(
            self.model_1.predict(x),
            index=x.index,
            columns=y.columns,
        )
        y_resid = y - y_fit
        self.model_2.fit(x, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns

        # Save data for possible checking
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, x):
        """Predict with the model."""
        y_pred1 = pd.DataFrame(
            self.model_1.predict(x),
            index=x.index,
            columns=self.y_columns,
        )
        y_pred2 = pd.DataFrame(
            self.model_2.predict(x),
            index=x.index,
            columns=self.y_columns,
        )
        y_pred = y_pred1 + y_pred2

        return y_pred
