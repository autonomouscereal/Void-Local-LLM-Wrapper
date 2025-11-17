#!/usr/bin/env bash
set -euo pipefail

IFS=',' read -ra REQS <<< "${REQUIRED_SERVICES:-}"
for s in "${REQS[@]:-}"; do
  [ -n "${s:-}" ] || continue
  echo "[gate] waiting for $s health..."
  case "$s" in
    music)    URL="${MUSIC_API_URL:-${MUSIC_FULL_API_URL:-http://127.0.0.1:7860}}/healthz" ;;
    xtts)     URL="${XTTS_API_URL:-http://127.0.0.1:8020}/healthz" ;;
    demucs)   URL="${DEMUCS_API_URL:-http://127.0.0.1:9101}/healthz" ;;
    film2)    URL="${FILM2_API_URL:-http://127.0.0.1:8090}/healthz" ;;
    melody)   URL="${MELODY_API_URL:-http://127.0.0.1:7861}/healthz" ;;
    dsinger)  URL="${SINGER_API_URL:-http://127.0.0.1:7862}/healthz" ;;
    rvc)      URL="${RVC_API_URL:-http://127.0.0.1:7863}/healthz" ;;
    vocalfix) URL="${VOCALFIX_API_URL:-http://127.0.0.1:7864}/healthz" ;;
    sfx)      URL="${SFX_API_URL:-http://127.0.0.1:7866}/healthz" ;;
    master)   URL="${MASTER_API_URL:-http://127.0.0.1:7865}/healthz" ;;
    mfa)      URL="${MFA_API_URL:-http://127.0.0.1:7867}/healthz" ;;
    prosody)  URL="${PROSODY_API_URL:-http://127.0.0.1:7868}/healthz" ;;
    *)        echo "[gate] unknown service $s"; exit 22 ;;
  esac
  ok=0
  for i in $(seq 1 120); do
    if curl -fsS "$URL" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" != "1" ]; then
    echo "[gate] $s not healthy, abort"; exit 10
  fi
done

echo "[gate] all required services healthy"

