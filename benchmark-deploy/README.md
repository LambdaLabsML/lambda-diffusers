
# Run the diffuser benchmark from a docker container

The following instructions show how to run the benchmarking program from a docker container on Ubuntu.

## Prerequisites

### Get a huggingface access token

Create a huggingface account.
Get a [huggingface access token](https://huggingface.co/docs/hub/security-tokens).

### Install NVIDIA docker

We first install docker:
```
# Install
sudo apt-get update
sudo apt-get remove docker docker-engine docker.io -y
sudo apt install containerd -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# Test install
docker --version

# Put the user in the docker group
sudo usermod -a -G docker $USER
newgrp docker
```

We install requirements to run docker containers leveraging NVIDIA GPUs:
```
# Install
sudo apt install curl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test install
sudo docker run --rm --gpus all nvidia/cuda:11.2.1-base-ubuntu20.04 nvidia-smi
```


3. Build the benchmark docker image

```
docker build -t benchmark -f ./benchmarking/Dockerfile .   
```

## Running the benchmark

Set the HuggingFace access token as environment variable:
```
export ACCESS_TOKEN=<your-hugging-face-access-token-here>
```

Run the benchmark program from the container and export the output `.csv` file to the host:
```
containerid=$(docker run --gpus all -e ACCESS_TOKEN=${ACCESS_TOKEN} -d --entrypoint "python3" benchmark:latest scripts/benchmark.py --samples=1,2,4,8,16) && \
docker wait ${containerid} && \
docker cp ${containerid}:/lambda_diffusers/benchmark_tmp.csv ./benchmark_tmp.csv
```

*Note that the arguments `scripts/benchmark.py --samples=1,2,4,8,16` can be changed to point to a different script or use different arguments.*