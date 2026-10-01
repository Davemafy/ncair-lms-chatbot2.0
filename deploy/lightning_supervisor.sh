#!/usr/bin/env bash
set -Eeuo pipefail

LIVE_DIR="${NCAIR_LIVE_DIR:-$HOME/ncair-lms-live}"
DEPLOY_BRANCH="${NCAIR_DEPLOY_BRANCH:-lightning-live}"
POLL_SECONDS="${NCAIR_DEPLOY_POLL_SECONDS:-60}"
RETRY_SECONDS="${NCAIR_DEPLOY_RETRY_SECONDS:-600}"
PORT="${PORT:-8000}"
STATE_DIR="${NCAIR_STATE_DIR:-$HOME/.local/state/ncair-lms}"

PID_FILE="$STATE_DIR/server.pid"
HEALTHY_SHA_FILE="$STATE_DIR/healthy.sha"
FAILED_SHA_FILE="$STATE_DIR/failed.sha"
FAILED_AT_FILE="$STATE_DIR/failed.at"
SERVER_LOG="$STATE_DIR/server.log"
LOCK_FILE="$STATE_DIR/supervisor.lock"

mkdir -p "$STATE_DIR"
cd "$LIVE_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

require_environment() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN is missing. Add it to Lightning Studio environment variables/secrets."
    return 1
  fi

  export NCAIR_DEFAULT_VERSION="${NCAIR_DEFAULT_VERSION:-v2}"
  export NATLAS_MODEL="${NATLAS_MODEL:-NCAIR1/N-ATLaS}"
  export NATLAS_DEVICE="${NATLAS_DEVICE:-auto}"
  export NATLAS_QUANTIZATION="${NATLAS_QUANTIZATION:-4bit}"
  export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
}

ensure_system_dependencies() {
  if command -v pdftoppm >/dev/null 2>&1 && command -v tesseract >/dev/null 2>&1; then
    return 0
  fi

  log "Installing Poppler/Tesseract system dependencies"
  sudo apt-get update
  sudo apt-get install -y poppler-utils tesseract-ocr
}

remote_sha() {
  git ls-remote origin "refs/heads/$DEPLOY_BRANCH" | awk 'NR == 1 {print $1}'
}

server_running() {
  if [[ ! -s "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null
}

health_ok() {
  curl --fail --silent --show-error     --max-time 5     "http://127.0.0.1:$PORT/api/health" >/dev/null
}

stop_server() {
  if server_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    log "Stopping server pid=$pid"
    kill "$pid" 2>/dev/null || true

    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done

    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
}

install_runtime() {
  log "Ensuring Python runtime dependencies"
  python -m pip install -e ".[runtime]" >/dev/null
}

start_server() {
  : >"$SERVER_LOG"
  log "Starting FastAPI on port $PORT"
  nohup bash "$LIVE_DIR/deploy/start_lightning.sh" >>"$SERVER_LOG" 2>&1 &
  echo "$!" >"$PID_FILE"

  for _ in {1..120}; do
    if health_ok; then
      return 0
    fi
    if ! server_running; then
      log "Server exited during startup"
      tail -n 80 "$SERVER_LOG" || true
      return 1
    fi
    sleep 1
  done

  log "Health endpoint did not become ready"
  tail -n 80 "$SERVER_LOG" || true
  return 1
}

warmup_v2() {
  log "Warming N-ATLaS + FAISS with a Hausa factual request"

  python - "$PORT" <<'PY'
import json
import sys
import urllib.request

port = sys.argv[1]
payload = json.dumps({"message": "A ina ofishin NCAIR yake a Abuja?"}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/api/v2/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=300) as response:
    body = json.loads(response.read().decode())

if body.get("language") != "hausa":
    raise SystemExit(f"warmup language mismatch: {body.get('language')!r}")
if body.get("tool") != "search_ncair_knowledge_base":
    raise SystemExit(f"warmup tool mismatch: {body.get('tool')!r}")
if not body.get("answer"):
    raise SystemExit("warmup returned an empty answer")
if not body.get("sources"):
    raise SystemExit("warmup returned no official evidence sources")

print("warmup ok")
PY
}

checkout_target() {
  local sha="$1"
  git fetch --quiet origin "$DEPLOY_BRANCH"
  git checkout --quiet --detach "$sha"
}

deploy_sha() {
  local target="$1"
  local rollback_sha="${2:-}"

  log "Deploying $target"
  stop_server
  checkout_target "$target"

  if ! install_runtime || ! start_server || ! warmup_v2; then
    log "Deployment failed for $target"
    if [[ -s "$SERVER_LOG" ]]; then
      cp "$SERVER_LOG" "$STATE_DIR/server-$target.failed.log" || true
    fi
    echo "$target" >"$FAILED_SHA_FILE"
    date +%s >"$FAILED_AT_FILE"
    stop_server

    if [[ -n "$rollback_sha" && "$rollback_sha" != "$target" ]]; then
      log "Rolling back to $rollback_sha"
      checkout_target "$rollback_sha"
      install_runtime || true
      if start_server && warmup_v2; then
        echo "$rollback_sha" >"$HEALTHY_SHA_FILE"
        log "Rollback healthy at $rollback_sha"
      else
        log "Rollback failed; inspect $SERVER_LOG"
      fi
    fi
    return 1
  fi

  echo "$target" >"$HEALTHY_SHA_FILE"
  rm -f "$FAILED_SHA_FILE" "$FAILED_AT_FILE"
  log "Deployment healthy at $target"
}

failed_recently() {
  local target="$1"
  [[ -s "$FAILED_SHA_FILE" && -s "$FAILED_AT_FILE" ]] || return 1
  [[ "$(cat "$FAILED_SHA_FILE")" == "$target" ]] || return 1

  local failed_at now
  failed_at="$(cat "$FAILED_AT_FILE")"
  now="$(date +%s)"
  (( now - failed_at < RETRY_SECONDS ))
}

main() {
  require_environment
  ensure_system_dependencies

  while true; do
    local target healthy
    target="$(remote_sha || true)"

    if [[ -z "$target" ]]; then
      log "Could not resolve origin/$DEPLOY_BRANCH; retrying"
      sleep "$POLL_SECONDS"
      continue
    fi

    healthy=""
    if [[ -s "$HEALTHY_SHA_FILE" ]]; then
      healthy="$(cat "$HEALTHY_SHA_FILE")"
    fi

    if [[ "$target" == "$healthy" ]] && server_running && health_ok; then
      sleep "$POLL_SECONDS"
      continue
    fi

    if failed_recently "$target"; then
      if [[ "$healthy" != "" ]] && (! server_running || ! health_ok); then
        log "Live server is down; restoring last healthy commit $healthy"
        deploy_sha "$healthy" "" || true
      fi
      sleep "$POLL_SECONDS"
      continue
    fi

    deploy_sha "$target" "$healthy" || true
    sleep "$POLL_SECONDS"
  done
}

main "$@"
