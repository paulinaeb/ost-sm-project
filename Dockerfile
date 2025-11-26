# dockerfile
# filepath: Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forecasting ./forecasting
COPY preprocessing ./preprocessing
COPY streaming ./streaming
COPY dynamic_dataset ./dynamic_dataset

ENV CASSANDRA_HOSTS=cassandra-dev \
    CASSANDRA_PORT=9042 \
    CASSANDRA_KEYSPACE=linkedin_jobs \
    CASSANDRA_TABLE=jobs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501').read()" || exit 1

CMD ["streamlit", "run", "forecasting/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]