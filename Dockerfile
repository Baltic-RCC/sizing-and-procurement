# Define base container image
FROM $URL/docker-hub/python:3.11-slim

# Setup ENV
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# ENV RUST_LOG=trace
ENV PIP_INDEX_URL=$URL/pypi/rcc-pypi/simple
ENV PIP_TRUSTED_HOST=$URL

# Trust certs
COPY $CERT
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Set working directory
WORKDIR /app

# Set entrypoint
# ENTRYPOINT ["sleep","infinity"]

# Install dependency management modules
# Install tools to set up python environment
RUN pip install --upgrade setuptools pip uv pipenv
# RUN pip install --upgrade pip uv

# Install python dependencies
COPY pyproject.toml uv.lock .
# RUN uv pip sync pyproject.toml --system && uv cache clean
# RUN uv pip sync pyproject.toml --system
# RUN uv sync && uv cache clean
RUN uv pip install . --system
# UN uv sync --system does not work

# Add application files to container
COPY py/common/ ./py/common/
COPY py/atc_data_transfer/ ./py/atc_data_transfer/
COPY py/data_classes/ ./py/data_classes/
COPY py/procurement/ ./py/procurement/
COPY py/handlers/ ./py/handlers/
COPY py/parsers/ ./py/parsers/
COPY resources/schemas/ ./resources/schemas/
COPY config/ ./config/

ENV PYTHONPATH="${PYTHONPATH}:/app"

ARG CMD_NAME="python py/procurement/procurement_from_elastic.py"
ENV ENV_CMD_NAME="$CMD_NAME"
CMD /bin/sh -c "$ENV_CMD_NAME"

