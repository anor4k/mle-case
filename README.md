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

#### Running with Docker

This project has a `pipeline.Dockerfile` and an `api.Dockerfile` for running the train
pipeline and serving the API, respectively, in a reproductible environment.

You can place the files `train.csv` and `test.csv` in the project directory and run the
`train_and_serve.sh` script to quickly build the images and run the pipeline and serve
the model.

To serve the model, run the image built from `api.Dockerfile` by binding the model file
to `/model.pkl` and the port `8000` to expose FastAPI.

### Training

To run the training pipeline, use `uv run src/property_pipeline_bain/pipeline.py --train-path /path/to/train.csv --test-path /path/to/test.csv`.

By default, this will save a `model.pkl` file in the current directory. To override the
output path, use the `--output-path /path/to/model.pkl` argument.

#### With Docker

When running the training pipeline, mount the train and test files to `/train.csv` and
`/test.csv` paths in the container, respectively, using
`--mount type-bind,source=/path/to/file.csv,target=/file.csv` to mount. You must also
specify the output model path, which will be read by the API container, to `/model.pkl`.

```shell
docker build -t bain_pipeline -f pipeline.Dockerfile .

docker run \
--mount type=bind,source="$(pwd)/train.csv",target=/train.csv \
--mount type=bind,source="$(pwd)/test.csv",target=/test.csv \
--mount type=bind,source="$(pwd)/model.pkl",target=/model.pkl \
bain_pipeline
```

### API

To run the FastAPI server, use `uv run fastapi run src/property_pipeline_bain/api.py`.
The API will be available at `localhost:8000`.

By default, it will try to load a `model.pkl` file from the current directory. If you
wish to override the model source, set the `PROPERTY_MODEL_PATH` environment variable
pointing to your trained model.

### Authentication

Authentication is done using a simple API key set via the `BAIN_API_KEY` environment
variable. In your request, you must pass the `Bain-API-Key: YOUR_KEY_HERE` header.

### Making a prediction

To make a prediction through the API, make a POST request to the `/predict` endpoint,
passing the feature information as a JSON body. All features are mandatory.

Example prediction:

```shell
curl --location 'localhost:8000/predict' \
--header 'Content-Type: application/json' \
--header 'Bain-API-Key: YOUR_KEY_HERE' \
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

#### Deploying with Docker

You need to bind the model file and expose the 8000 port for FastAPI. To configure an
API key, set the `BAIN_API_KEY` variable in your .env file.

```shell
docker build -t bain_api -f api.Dockerfile .

docker run \
--mount type=bind,source="$(pwd)/model.pkl",target=/model.pkl \
-p 8000:8000 \
bain_api
```

### Documentation

API documentation can be accessed via the `/docs` endpoint.

## Development

### Assumptions

Though mostly modularized, the model's parameters and preprocessing steps are hardcoded.
This project is only targeted at training a final model and serving it through an API,
but it could benefit from a more customizable approach if the goal is to serve different
or multiple models.

While Pydantic is used to validate data input to the API, no checks are made about the
quality of data. Abnormal inputs will lead to abnormal outputs.

### Next steps and improvements

A complete test suite is warranted for production, to make sure changes don't break
expected behavior.

This code could be easily deployed for recurrent re-training using simple cron jobs or
airflow, by simply updating the .csv files and running the cli command again. Since db
connections are not yet implemented, the CLI would also need to be adjusted.

Currently, training parameters and test metrics are only printed to `stdout`. A better
approach would be to use MLflow to log metrics and evaluate model performance over
time. MLflow could also be used to log models, and even to serve models for the FastAPI
server or through its own framework.

Another improvement opportunity is auto-parameter tuning, which would be useful in
recurrent training, since they are currently hardcoded. Using a framework like Optuna
is ideal, though a simple solution like GridSearchCV is also viable. MLflow can help
with logging all the hyperparameter runs and choosing the best model.

For production, CI through GitHub Actions would enable automatic code checks and
automatic deployment of the server wherever it's hosted. `ruff` was used for basic
linting and formatting, but should be enforced through pre-commit. A docker compose file
for serving the API is also probably a better choice than bare `docker run`.
