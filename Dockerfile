FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root
WORKDIR /build
RUN python -m venv --without-pip /venv
COPY requirements.txt .
RUN pip --python /venv/bin/python install --no-cache-dir -r requirements.txt \
    && pip --python /venv/bin/python install --no-cache-dir 'msgpack>=1.2.1'

FROM builder AS test
COPY . .
RUN pip --python /venv/bin/python install --no-cache-dir -e '.[dev]'

FROM cgr.dev/chainguard/python:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/venv/bin:$PATH \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY --from=builder /venv /venv
COPY --chown=10001:10001 . .

USER 10001:10001
EXPOSE 8501
ENTRYPOINT ["python"]
CMD ["-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
