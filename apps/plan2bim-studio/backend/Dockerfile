FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/dajoong
COPY modules/plan2bim /srv/dajoong/modules/plan2bim
COPY apps/plan2bim-studio/backend /srv/dajoong/apps/plan2bim-studio/backend
RUN python -m pip install --upgrade pip && \
    python -m pip install /srv/dajoong/modules/plan2bim "/srv/dajoong/apps/plan2bim-studio/backend[aws,auth]"

RUN useradd --create-home --uid 10001 dajoong && chown -R dajoong:dajoong /srv/dajoong
USER dajoong
EXPOSE 8042
CMD ["uvicorn", "buili_plan2bim_studio.main:app", "--host", "0.0.0.0", "--port", "8042", "--proxy-headers"]
