import argparse
from pathlib import Path
from typing import overload

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline

MODEL_FEATURES = [
    "type",
    "sector",
    "net_usable_area",
    "net_area",
    "n_rooms",
    "n_bathroom",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = ["type", "sector"]

MODEL_TARGET = "price"


# This should probably be in a separate file, but we can keep it here for now.
def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate the mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate the root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate the mean absolute percentage error."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


class DataLoader:
    """Load data for training of the Property model.

    The class can be directly instantiated by providing train and test pandas
    dataframes, or by using the method for each appropriate source.
    For now, only CSV files are supported.
    """

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame | None = None):
        """Create a new instance of the DataLoader class.

        Though possible to instantiate directly, it is recommended to use one of the
        class methods to load the data from the appropriate source.

        Args:
            train_data (pd.DataFrame): The training data.
            test_data (pd.DataFrame, optional): The test data. Defaults to None.
"""
        self.train_data, self.test_data = self.validate_data(
            train_data=train_data, test_data=test_data
        )

    @overload
    @staticmethod
    def validate_data(
        train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    @staticmethod
    def validate_data(
        train_data: pd.DataFrame, test_data: None = ...
    ) -> tuple[pd.DataFrame, None]: ...

    @staticmethod
    def validate_data(
        train_data: pd.DataFrame, test_data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Validate the data before training the model."""

        if train_data.isna().any().any():
            raise ValueError("The training data contains missing values.")
        if test_data is not None and test_data.isna().any().any():
            raise ValueError("The test data contains missing values.")

        cols = [*MODEL_FEATURES, MODEL_TARGET]
        if not set(cols).issubset(train_data.columns):
            raise ValueError("The training data does not contain the required columns.")

        if test_data is not None and not set(cols).issubset(test_data.columns):
            raise ValueError("The test data does not contain the required columns.")

        train_data = train_data[cols]
        if test_data is not None:
            test_data = test_data[cols]

        return train_data, test_data

    def train_test_split(
        self, target: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
        """Return the data in the format required for training the model.

        If the test data was not provided during initialization, the method will return
        None for X_test and y_test.

        Returns:
            tuple: X_train, y_train, X_test, y_test
        """
        X_train = self.train_data.drop(target, axis=1)
        y_train = self.train_data[target]

        if self.test_data is not None:
            X_test = self.test_data.drop(target, axis=1)
            y_test = self.test_data[target]
        else:
            X_test, y_test = None, None

        return X_train, y_train, X_test, y_test

    @classmethod
    def from_csv(cls, train_path: str, test_path: str | None = None):
        """Load data from CSV files.

        Args:
            train_path (str): Path to the training data.
            test_path (str, optional): Path to the test data. Defaults to None."""
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path) if test_path else None

        return cls(train, test)

    @classmethod
    def from_database(cls, connection_string: str, query: str):
        # This class should query the database and return the dataframes.
        raise NotImplementedError


class ModelPipeline:
    """Create a pipeline for training and prediction of a Property model.

    The class can be directly instantiated by providing the DataLoader object,
    or by using the method for each appropriate source. To instantiate from a trained
    model, use the `load_pipeline` class method.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Args:
            data_loader (DataLoader): An instance of the DataLoader class, loaded
                with the train data and optionally test data."""
        self.data_loader = data_loader
        # Model params are hardcoded here but could be loaded from a config file
        # or passed as arguments in the constructor.
        self.pipeline = self._create_pipeline(
            categorical_features=CATEGORICAL_FEATURES,
            learning_rate=0.01,
            n_estimators=300,
            max_depth=5,
            loss="absolute_error",
        )

    def _create_pipeline(self, categorical_features: list[str], **model_params):
        """Create a pipeline for training the model."""

        # Define preprocessing
        column_transformer = ColumnTransformer(
            transformers=[("categorical", TargetEncoder(), categorical_features)]
        )

        # Define the model
        model = GradientBoostingRegressor(**model_params)

        # Define the pipeline
        pipeline = Pipeline(
            steps=[("preprocessor", column_transformer), ("model", model)]
        )

        return pipeline

    def train(self) -> Pipeline:
        """Train the model using the pipeline."""

        X_train, y_train, X_test, y_test = self.data_loader.train_test_split(
            target=MODEL_TARGET
        )
        self.pipeline.fit(X_train, y_train)

        if X_test is not None and y_test is not None:
            y_pred = self.pipeline.predict(X_test)
            mape_test = mape(y_test, y_pred)
            rmse_test = rmse(y_test, y_pred)
            mae_test = mae(y_test, y_pred)

            # we could log these to mlflow
            print(f"MAPE: {mape_test:6>.2%}")
            print(f"RMSE: {rmse_test:6>.2f}")
            print(f"MAE:  {mae_test:6>.2f}")

        return self.pipeline

    def save_pipeline(self, path: str) -> None:
        """Save the pipeline to a file."""
        if self.pipeline is None:
            raise ValueError("The pipeline has not been trained yet.")

        # if using mlflow, use PythonModel and save with pyfunc.log_model
        joblib.dump(self.pipeline, Path(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=False, default=None)
    parser.add_argument("--output-path", type=str, required=False, default="model.pkl")

    # We could add arguments here for the database connection when it's implemented,
    # and require either .csv files or a database connection for training.

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)

    data_loader = DataLoader(train_data, test_data)
    model_pipeline = ModelPipeline(data_loader)

    model_pipeline.train()
    model_pipeline.save_pipeline(args.output_path)
