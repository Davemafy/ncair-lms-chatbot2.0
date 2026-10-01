#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIGHTNING_DIR="$HOME/.lightning_studio"
STATE_DIR="${NCAIR_STATE_DIR:-$HOME/.local/state/ncair-lms}"
LIVE_DIR="${NCAIR_LIVE_DIR:-$HOME/ncair-lms-live}"
DEPLOY_BRANCH="${NCAIR_DEPLOY_BRANCH:-lightning-live}"
REPO_URL="${NCAIR_REPO_URL:-https://github.com/Davemafy/ncair-lms-chatbot2.0.git}"

mkdir -p "$LIGHTNING_DIR" "$STATE_DIR"

install -m 0755   "$REPO_ROOT/deploy/lightning_on_start.sh"   "$LIGHTNING_DIR/on_start.sh"

if [[ ! -d "$LIVE_DIR/.git" ]]; then
  git clone --branch "$DEPLOY_BRANCH" --single-branch "$REPO_URL" "$LIVE_DIR"
else
  git -C "$LIVE_DIR" remote set-url origin "$REPO_URL"
  git -C "$LIVE_DIR" fetch origin "$DEPLOY_BRANCH"
fi

if ! command -v pdftoppm >/dev/null 2>&1 || ! command -v tesseract >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y poppler-utils tesseract-ocr
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  cat <<'EOF'
WARNING: HF_TOKEN is not visible in this shell.
Add HF_TOKEN to Lightning Studio -> Environment variables (or Teamspace Secrets)
before relying on automatic restarts.
EOF
fi

# Hand port 8000 over to the managed supervisor if a manually started
# instance is still running.
pkill -f 'uvicorn ncair_lms.api:app.*--port 8000' 2>/dev/null || true
sleep 1

bash "$LIGHTNING_DIR/on_start.sh"

echo
echo "Lightning automation installed."
echo "Startup hook: $LIGHTNING_DIR/on_start.sh"
echo "Live checkout: $LIVE_DIR"
echo "State/logs: $STATE_DIR"
echo
echo "Watch deployment:"
echo "  tail -f $STATE_DIR/supervisor.log"
echo
echo "Watch API:"
echo "  tail -f $STATE_DIR/server.log"
