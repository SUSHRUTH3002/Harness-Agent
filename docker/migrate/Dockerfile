# Image that installs the harness with the Postgres extra and creates its tables.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Only what the package build needs (hatchling reads README.md for the metadata).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]"

COPY docker/migrate/create_tables.py ./create_tables.py
CMD ["python", "create_tables.py"]
