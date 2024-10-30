ARG IMAGE=pytorch/pytorch
ARG TAG=latest
ARG BASE=pytorch-cpu

FROM ubuntu:22.04 AS pytorch-cpu

ARG PYTHON_VERSION=3.11
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev python3-pip python${PYTHON_VERSION} && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

FROM ${IMAGE}:${TAG} as pytorch

FROM ${BASE}
RUN adduser --no-create-home --home /opt/aiops aiops
RUN mkdir -p /opt/aiops; chown aiops:aiops /opt/aiops

WORKDIR /opt/aiops
USER aiops

COPY . /opt/aiops

ARG MODEL_PATH="./models/IRIS-BERT-base-Categorizer/"
ENV MODEL_PATH="${MODEL_PATH}"

ENV PATH=${PATH}:/opt/aiops/.local/bin
RUN pip install -r requirements.txt

EXPOSE 8000
CMD fastapi run api.py
