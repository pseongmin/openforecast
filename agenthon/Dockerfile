# openforecast — Agenthon 2026 Track 2 agent image (linux/amd64)
FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "numpy==2.1.3" "pandas==2.2.3" "pyarrow==18.1.0" "jsonschema==4.23.0" \
    "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.2.tar.gz#subdirectory=common"
WORKDIR /work
COPY qfagent /opt/qfagent
ENV PYTHONPATH=/opt
RUN printf '#!/bin/sh\nexec python3 -m qfagent.cli "$@"\n' > /usr/local/bin/forecast && chmod +x /usr/local/bin/forecast
RUN useradd --create-home --uid 1000 runner
USER runner
CMD ["forecast", "--help"]
