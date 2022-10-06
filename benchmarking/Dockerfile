FROM nvidia/cuda:11.2.1-base-ubuntu20.04
RUN \
    apt-get -y update \
    && apt-get -y install -y git python3 python3-pip \
    && git clone https://github.com/LambdaLabsML/lambda-diffusers.git \
    && cd lambda-diffusers \
    && pip install -r requirements.txt
CMD ["python3", "-u", "/lambda-diffusers/scripts/benchmark.py"]