#!/bin/bash

# Build benchmark docker image and save as .tar file
docker build -t benchmark -f ./Dockerfile .   
docker save -o ./benchmark.tar benchmark:latest