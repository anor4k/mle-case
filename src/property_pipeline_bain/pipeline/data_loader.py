from typing import overload

import pandas as pd

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
