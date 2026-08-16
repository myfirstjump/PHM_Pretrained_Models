# timesfm>=2.5 requires Python >= 3.12
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    MPLCONFIGDIR=/tmp/mpl

# fonts-noto-cjk: Traditional Chinese labels in matplotlib figures
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-models.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt -r requirements-models.txt

COPY main.py ./
COPY src/ ./src/

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
