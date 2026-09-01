FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 satquery

WORKDIR /srv/satquery
COPY backend ./backend
COPY ml ./ml
COPY model_service ./model_service
RUN python3.11 -m pip install ./backend ./ml ./model_service \
    && python3.11 -m pip install -r ml/requirements-notebook.txt

USER satquery
EXPOSE 8080
CMD ["python3.11", "-m", "uvicorn", "satquery_model_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
