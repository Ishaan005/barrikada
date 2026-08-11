# syntax=docker/dockerfile:1

# Build arguments used by FROM must be declared before the first stage so they
# remain available when Docker parses later stage declarations.
ARG JENTIC_BASE=ghcr.io/jentic/jentic-one-app@sha256:9a558e431ee814e60109720fb44466582098681578fc039b5c07d57150ec55bd

FROM python:3.12-slim AS plugin-builder

WORKDIR /build
COPY integrations/jentic-one /build/integrations/jentic-one
RUN python -m pip wheel --no-deps --wheel-dir /wheels /build/integrations/jentic-one

# Jentic 0.29.1 official release image, pinned to its published immutable digest.
FROM ${JENTIC_BASE}

ARG BARRIKADE_VERSION=0.2.0
USER root
COPY --from=plugin-builder /wheels/barrikade_jentic-*.whl /tmp/
RUN python -m pip install --no-cache-dir --no-deps /tmp/barrikade_jentic-*.whl \
    && rm /tmp/barrikade_jentic-*.whl

LABEL org.opencontainers.image.title="Barrikade-enabled Jentic One" \
      org.opencontainers.image.version="${BARRIKADE_VERSION}" \
      io.barrikade.jentic.version="0.29.1" \
      io.barrikade.api.version="2"

USER jentic
CMD ["python", "-m", "barrikade_jentic.bootstrap"]
