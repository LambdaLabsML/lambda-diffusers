#!/bin/bash

# Load the benchmark image from .tar file into Docker
docker load -i ./benchmark.tar

# Run benchmark
# 1. run the container
# 2. wait for benchmark completion
# 3. export benchmark output to host`
docker build -t benchmark -f ./Dockerfile . && \
printf "\nRunning the diffusers benchmark..." && \
containerid=$(docker run --gpus all -e ACCESS_TOKEN=${ACCESS_TOKEN} -t -d benchmark) && \
docker wait ${containerid} && \
docker cp \
  ${containerid}:/lambda-diffusers/benchmark_tmp.csv ./benchmark_tmp.csv && \
printf "\nDiffuser benchmark complete! -> [benchmark_tmp.csv]"
