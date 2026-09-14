FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home ceph
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=ceph:ceph app ./app
COPY --chown=ceph:ceph models ./models
USER ceph
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log --timeout-keep-alive 5"]
