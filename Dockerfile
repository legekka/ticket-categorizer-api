ARG IMAGE=pytorch/pytorch
ARG TAG=2.4.1-cuda12.4-cudnn9-runtime

FROM ${IMAGE}:${TAG} AS base

RUN apt-get update \
    && apt-get -y install libpq-dev gcc

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
