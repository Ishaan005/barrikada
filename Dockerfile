FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.fast.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.fast.txt


FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    BARRIKADE_CORE_MODELS_DIR=/models \
    BARRIKADE_ACTIVE_PROFILE=jentic_gateway_fast \
    BARRIKADE_AUTO_MIGRATE=false \
    BARRIKADE_ALLOW_DIAGNOSTICS=false \
    BARRIKADE_VERIFY_ARTIFACTS=true \
    BARRIKADE_BUNDLE_MANIFEST_PATH=/models/manifest.json \
    BARRIKADE_BUNDLE_PUBLIC_KEY_PATH=/etc/barrikade/bundle-public.pem

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin barrikade \
    && mkdir -p /models /etc/barrikade /var/lib/barrikade \
    && chown -R barrikade:barrikade /app /models /etc/barrikade /var/lib/barrikade

COPY --from=builder /install /usr/local

COPY api /app/api
COPY models /app/models
COPY barrikade /app/barrikade
COPY core/__init__.py core/__version__.py core/artifacts.py core/incident_reporter.py \
    core/intent_scorer.py core/onnx_encoder.py core/orchestrator.py core/risk_budget.py \
    core/onnx_parity.py \
    core/session.py core/session_orchestrator.py core/session_settings.py core/settings.py \
    core/telemetry.py core/profile_routing.py core/release_policy.py /app/core/
COPY core/layer_a /app/core/layer_a
COPY core/layer_b/__init__.py core/layer_b/signature_engine.py /app/core/layer_b/
COPY core/layer_c/__init__.py core/layer_c/classifier.py /app/core/layer_c/
COPY core/layer_d/__init__.py core/layer_d/classifier.py /app/core/layer_d/

RUN chown -R barrikade:barrikade /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=5)"

USER barrikade

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
