# Property Price Prediction model - Bain MLE Case

This repo contains files for Bain's recruitment case for a MLE position.

This project's goal is to define a training pipeline for a Property Price prediction
model, allowing the model to be trained from either .csv files, or a db connection,
though the latter is not currently implemented.

## Usage

### Installation

Clone this repo and `cd` into it. To install dependencies, [`uv`](https://astral.sh/uv)
is recommended. To install `uv`, run `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Run `uv venv` to automatically create a virtual environment, and `uv sync` to install
the dependencies from `pyproject.toml`.

### Training

To run the training pipeline, use `uv run src/property_pipeline_bain/pipeline.py --train-path /path/to/train.csv --test-path /path/to/test.csv`.

By default, this will save a `model.pkl` file in the current directory. To override the
output path, use the `--output-path /path/to/model.pkl` argument.

### API

#### Running

To run the FastAPI server, use `uv run fastapi run src/property_pipeline_bain/api.py`.
The API will be available at `localhost:8000`, by default.

By default, it will try to load a `model.pkl` file from the current directory. If you
wish to override the model source, set the `PROPERTY_MODEL_PATH` environment variable
pointing to your trained model.

#### Making a prediction

To make a prediction through the API, make a POST request to the `/predict` endpoint,
passing the feature information as a JSON body. All features are mandatory.

Example prediction:

```shell
curl --location 'localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "type": "departamento",
    "sector": "vitacura",
    "net_usable_area": 140.0,
    "net_area": 170.0,
    "n_rooms": 4.0,
    "n_bathroom": 4.0,
    "latitude": -33.40123,
    "longitude": -70.5805
}'
```

Which returns: `13070.575463557223` (results may vary depending on the training data and
other factors).

## Development

### Assumptions

Though mostly modularized, the model's parameters and preprocessing steps are hardcoded.
This project is only targeted at training a final model and serving it through an API,
but it could benefit from a more customizable approach if the goal is to serve different
or multiple models.

While Pydantic is used to validate data input to the API, no checks are made about the
quality of data. Abnormal inputs will lead to abnormal outputs.

### Next steps and improvements

This project has very basic tests configured using `pytest`. A more complete test suite
is warranted for production, to make sure changes don't break expected behavior.

This code could be easily deployed for recurrent re-training using simple cron jobs or
airflow, by simply updating the .csv files and running the cli command again. Since db
connections are not yet implemented, the CLI would also need to be adjusted.

Currently, training parameters and test metrics are only printed to `stdout`. A better
approach would be to use MLflow to log metrics and evaluate model performance over
time. MLflow could also be used to log models, and even to serve models for the FastAPI
server or through its own framework.
