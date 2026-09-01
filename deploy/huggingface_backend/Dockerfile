FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SATQUERY_HOST=0.0.0.0 \
    SATQUERY_PORT=7860 \
    SATQUERY_DATA_DIR=/srv/satquery/data/runtime \
    SATQUERY_DATABASE_PATH=/srv/satquery/data/runtime/satquery.sqlite3 \
    SATQUERY_UPLOAD_DIR=/srv/satquery/data/runtime/uploads \
    SATQUERY_ARTIFACT_DIR=/srv/satquery/data/runtime/artifacts \
    SATQUERY_REPORT_DIR=/srv/satquery/data/runtime/reports

RUN useradd --create-home --uid 1000 satquery
WORKDIR /srv/satquery

COPY requirements.txt ./requirements.txt
RUN pip install --requirement requirements.txt

COPY app ./app
RUN mkdir -p /srv/satquery/data/runtime && chown -R satquery:satquery /srv/satquery

USER satquery
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]

