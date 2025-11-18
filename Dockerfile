FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY . .

ENTRYPOINT ["bsort"]

CMD ["--help"]