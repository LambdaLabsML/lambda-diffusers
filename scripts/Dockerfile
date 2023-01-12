FROM nvcr.io/nvidia/pytorch:22.11-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --pre xformers
RUN pip install diffusers==0.11.0 accelerate transformers

WORKDIR /workspace

COPY benchmark.py /workspace/benchmark.py
RUN (printf '#!/bin/bash\npython benchmark.py \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]