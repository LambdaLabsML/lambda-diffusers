#!/bin/bash

# Installing Docker and The Docker Utility Engine for NVIDIA GPUs
# cf :https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html

# Docker (sometimes rebooting system is necessary for this to work...)
sudo apt-get update
sudo apt-get remove docker docker-engine docker.io -y
sudo apt install containerd -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
docker --version

# Put the user in the docker group
sudo usermod -a -G docker $USER
newgrp docker

# Nvidia Docker
sudo apt install curl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test Nvidia Docker
# sudo docker run --rm --gpus all nvidia/cuda:11.2.1-base-ubuntu20.04 nvidia-smi

## Erase all Docker images [!!! CAUTION !!!]
# docker rmi -f $(docker images -a -q)
