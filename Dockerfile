# CPU-only, ONNX Runtime image for the dashboard. No PyTorch/Ultralytics inside (smaller image,
# and no AGPL code bundled). Mount ONNX models at /app/models; they are not baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SHELFSIGHT_ROOT=/app
WORKDIR /app

# ffmpeg is optional: lets the dashboard re-encode annotated video to browser-playable H.264.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN pip install --no-deps -e .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models /app/reports /app/data \
    && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/models", "/app/reports"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
