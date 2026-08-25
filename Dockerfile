FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml config.toml ./
COPY app.py config.py errors.py lanes.py ./
COPY inference ./inference
COPY routers ./routers
COPY prompts ./prompts

RUN pip install --no-cache-dir -e .

# Listen inside the container. Compose owns networks and whether this is reachable.
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
