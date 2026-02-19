FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml /app/
COPY src /app/src
COPY scripts /app/scripts

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "clinical_nlp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
