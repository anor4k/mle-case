#!/bin/bash
docker build -t bain_pipeline -f pipeline.Dockerfile .

docker build -t bain_api -f api.Dockerfile .

# Train the model
docker run \
--mount type=bind,source="$(pwd)/train.csv",target=/train.csv \
--mount type=bind,source="$(pwd)/test.csv",target=/test.csv \
--mount type=bind,source="$(pwd)/model.pkl",target=/model.pkl \
bain_pipeline

# Serve the model
docker run \
--mount type=bind,source="$(pwd)/model.pkl",target=/model.pkl \
-p 8000:8000 \
bain_api
