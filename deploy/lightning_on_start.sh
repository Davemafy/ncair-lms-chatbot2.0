#!/usr/bin/env bash
set -u

REPO_URL="${NCAIR_REPO_URL:-https://github.com/Davemafy/ncair-lms-chatbot2.0.git}"
LIVE_DIR="${NCAIR_LIVE_DIR:-$HOME/ncair-lms-live}"
DEPLOY_BRANCH="${NCAIR_DEPLOY_BRANCH:-lightning-live}"
STATE_DIR="${NCAIR_STATE_DIR:-$HOME/.local/state/ncair-lms}"

mkdir -p "$STATE_DIR"
exec >>"$STATE_DIR/on-start.log" 2>&1

echo "[$(date -Is)] Lightning start hook"

if [[ ! -d "$LIVE_DIR/.git" ]]; then
  git clone --branch "$DEPLOY_BRANCH" --single-branch "$REPO_URL" "$LIVE_DIR"
fi

cd "$LIVE_DIR"
git remote set-url origin "$REPO_URL"
git fetch --quiet origin "$DEPLOY_BRANCH" || true

nohup bash "$LIVE_DIR/deploy/lightning_supervisor.sh" \
  >>"$STATE_DIR/supervisor.log" 2>&1 &

echo "[$(date -Is)] supervisor launch requested pid=$!"
