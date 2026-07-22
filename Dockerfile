FROM python:latest

RUN apt-get update -y && \
    apt-get install python3-opencv -y

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev
COPY src/ ./src/
ENV PYTHONPATH=/app/src
CMD ["uv", "run", "python", "-m", "src.detector_neumonia"]
