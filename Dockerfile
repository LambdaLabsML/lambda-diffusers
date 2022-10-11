FROM nvidia/cuda:11.2.1-base-ubuntu20.04
RUN apt-get update && \
    apt-get install --no-install-recommends --no-install-suggests -y \
    curl python3 python3-pip
WORKDIR /lambda_diffusers
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["python3", "-u", "scripts/benchmark.py", "--samples", "1,2,4,8,16"]