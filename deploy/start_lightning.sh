#!/usr/bin/env bash
set -euo pipefail

export NCAIR_DEFAULT_VERSION="${NCAIR_DEFAULT_VERSION:-v2}"
export NATLAS_MODEL="${NATLAS_MODEL:-NCAIR1/N-ATLaS}"
export NATLAS_DEVICE="${NATLAS_DEVICE:-auto}"

exec uvicorn ncair_lms.api:app --host 0.0.0.0 --port "${PORT:-8000}"
