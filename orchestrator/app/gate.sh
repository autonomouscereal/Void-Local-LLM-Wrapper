#!/usr/bin/env bash
set -euo pipefail

IFS=',' read -ra REQS <<< "${REQUIRED_SERVICES:-}"
for s in "${REQS[@]:-}"; do
  [ -n "${s:-}" ] || continue
  echo "[gate] waiting for $s health..."
  case "$s" in
    music)  URL="http://music:7860/healthz" ;;
    yue)    URL="${YUE_API_URL:-http://yue:9001}/healthz" ;;
    xtts)   URL="${XTTS_API_URL:-http://xtts:8020}/healthz" ;;
    demucs) URL="http://demucs:9101/healthz" ;;
    film2)  URL="http://film2:8090/healthz" ;;
    melody) URL="http://melody:7861/healthz" ;;
    dsinger) URL="http://dsinger:7862/healthz" ;;
    rvc)    URL="http://rvc:7863/healthz" ;;
    vocalfix) URL="http://vocalfix:7864/healthz" ;;
    sfx)    URL="http://sfx:7866/healthz" ;;
    master)  URL="http://master:7865/healthz" ;;
    mfa)     URL="http://mfa:7867/healthz" ;;
    prosody) URL="http://prosody:7868/healthz" ;;
    *)      echo "[gate] unknown service $s"; exit 22 ;;
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

