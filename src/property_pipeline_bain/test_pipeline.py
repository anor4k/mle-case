import pandas as pd
import pytest
from sklearn.base import check_is_fitted

from property_pipeline_bain.pipeline import DataLoader, ModelPipeline


@pytest.fixture
def mock_data():
    # This data is not real and was manually inputted for testing purposes only.
    mock_train = pd.DataFrame(
        data=[
            {
                "type": "mock_type_1",
                "sector": "mock_sector_1",
                "net_usable_area": 160.0,
                "net_area": 190.0,
                "n_rooms": 4.0,
                "n_bathroom": 3.0,
                "latitude": -33.41000,
                "longitude": -70.59000,
                "price": 12500,
            },
            {
                "type": "mock_type_2",
                "sector": "mock_sector_1",
                "net_usable_area": 240.0,
                "net_area": 680.0,
                "n_rooms": 6.0,
                "n_bathroom": 5.0,
                "latitude": -33.45000,
                "longitude": -70.58000,
                "price": 18500,
            },
            {
                "type": "mock_type_2",
                "sector": "mock_sector_2",
                "net_usable_area": 125.0,
                "net_area": 220.0,
                "n_rooms": 5.0,
                "n_bathroom": 4.0,
                "latitude": -33.40000,
                "longitude": -70.56000,
                "price": 11000,
            },
        ]
    )

    mock_test = pd.DataFrame(
        [
            {
                "type": "mock_type_1",
                "sector": "mock_sector_2",
                "net_usable_area": 270.0,
                "net_area": 270.0,
                "n_rooms": 5.0,
                "n_bathroom": 6.0,
                "latitude": -33.36000,
                "longitude": -70.55000,
                "price": 32000,
            },
            {
                "type": "mock_type_2",
                "sector": "mock_sector_1",
                "net_usable_area": 85.0,
                "net_area": 95.0,
                "n_rooms": 4.0,
                "n_bathroom": 3.0,
                "latitude": -33.45000,
                "longitude": -70.62000,
                "price": 6000,
            },
        ]
    )

    return mock_train, mock_test


def test_train_pipeline(mock_data):
    mock_train, mock_test = mock_data
    data_loader = DataLoader(mock_train, mock_test)
    train_pipeline = ModelPipeline(data_loader)

    train_pipeline.train()

    assert train_pipeline.pipeline is not None
    # check_is_fitted will return None if the pipeline is fitted
    assert check_is_fitted(train_pipeline.pipeline) is None
