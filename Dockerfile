FROM public.ecr.aws/amazonlinux/amazonlinux:2023-minimal

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/vectory \
    XDG_CACHE_HOME=/home/vectory/.cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN microdnf upgrade -y \
    && microdnf install -y python3.11 python3.11-pip shadow-utils \
    && microdnf clean all \
    && python3.11 -m venv /opt/venv

COPY requirements.txt .
COPY infra/vulnerability-maintenance/patches/requirements.txt /tmp/security-patches.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt -r /tmp/security-patches.txt \
    && pip check \
    && pip uninstall --yes pip

RUN groupadd --system --gid 10001 vectory \
    && useradd --system --uid 10001 --gid vectory --home-dir /home/vectory --create-home vectory \
    && chown vectory:vectory /app

COPY --chown=10001:10001 . .

USER 10001:10001

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
