FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 satquery
WORKDIR /srv/satquery

COPY requirements.txt ./requirements.txt
RUN pip install --requirement requirements.txt

COPY app ./app
RUN mkdir -p /srv/satquery/data/runtime && chown -R satquery:satquery /srv/satquery

USER satquery
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

