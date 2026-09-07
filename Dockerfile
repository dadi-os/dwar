FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml config.toml ./
COPY app.py config.py errors.py lanes.py logutil.py ./
COPY inference ./inference
COPY routers ./routers
COPY prompts ./prompts

RUN pip install --no-cache-dir -e .

EXPOSE 8080

FROM base AS production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

FROM base AS dev
# Source arrives via Nas bind mount; site-packages stay in the image.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
