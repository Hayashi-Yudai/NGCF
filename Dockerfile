FROM nvcr.io/nvidia/pytorch:22.02-py3

WORKDIR /workdir

ENV PATH $PATH:/root/.local/bin

COPY pyproject.toml poetry.lock /workdir/

RUN apt-get update \
    && (curl -sSL https://install.python-poetry.org | python3 -) \
    && poetry install --no-root