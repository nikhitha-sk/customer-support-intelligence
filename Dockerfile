FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY csi ./csi
COPY dashboard ./dashboard
COPY data/knowledge_base ./data/knowledge_base
COPY reports ./reports
# Models and indexes are produced by the pipeline (see README) and mounted at runtime.
VOLUME ["/app/artifacts", "/app/data/processed"]
EXPOSE 8000 8501
CMD ["uvicorn", "csi.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
