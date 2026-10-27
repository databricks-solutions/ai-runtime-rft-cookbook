#!/usr/bin/env bash
# Isolate SGLang shutdown on one GPU. Child failures are intentional; setup failures are not.
# MODE: clean (server SIGTERM), crash/nodiag (detokenizer SIGTERM), group (whole-group SIGTERM).
set -euo pipefail
export PATH=/opt/venv/bin:$PATH
: "${MODE:?set MODE}" "${MODEL:?set MODEL}"
case "$MODE" in clean|crash|nodiag|group) ;; *) echo "unknown MODE: $MODE" >&2; exit 2 ;; esac
PORT=${PORT:-30000}
LOG_DIR=${LOG_DIR:-$(mktemp -d /tmp/rft-sglang-stop.XXXXXX)}
mkdir -p "$LOG_DIR"
SERVER=
cleanup() {
  # Only the session this script started. Never match processes by a broad command-line pattern.
  if [[ -n "$SERVER" && "$SERVER" -gt 1 ]]; then
    kill -KILL -- "-$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[debug] mode=$MODE host=$(hostname) logs=$LOG_DIR"
python -c 'import sglang; print("[debug] sglang", sglang.__version__)'
if [[ -f /opt/venv/stack.json ]]; then
  python -c 'from pathlib import Path; print(Path("/opt/venv/stack.json").read_text())'
fi
if [[ "$MODE" == nodiag ]]; then
  export SGLANG_PYSPY_DUMP_BEFORE_CRASH=0 SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
fi
setsid python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --tp 1 \
  --mem-fraction-static 0.6 --attention-backend triton > "$LOG_DIR/sglang.log" 2>&1 &
SERVER=$!
READY=0
for ((i=0; i<${HEALTH_ATTEMPTS:-120}; i++)); do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    tail -n 60 "$LOG_DIR/sglang.log" >&2
    echo "[debug] server exited before readiness" >&2
    exit 1
  fi
  if curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep "${HEALTH_INTERVAL_S:-5}"
done
if [[ "$READY" != 1 ]]; then echo "[debug] server readiness timed out" >&2; exit 1; fi
curl -fsS --connect-timeout 2 --max-time 60 "http://127.0.0.1:$PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8}}' > "$LOG_DIR/generation.json"
python -c 'import json, sys; r=json.load(open(sys.argv[1])); assert "text" in r and "error" not in r, r' "$LOG_DIR/generation.json"
echo "[debug] readiness and generation passed"
echo "[debug] processes before the stop:"
ps -eo pid,ppid,pgid,stat,comm,args --forest | cut -c1-180
case "$MODE" in
  crash|nodiag)
    mapfile -t DETOKENIZERS < <(ps -eo pid=,pgid=,args= | awk -v group="$SERVER" \
      '$2 == group && $3 ~ /^sglang::detokenizer/ { print $1 }')
    if [[ ${#DETOKENIZERS[@]} != 1 ]]; then
      echo "[debug] expected one owned detokenizer, found ${#DETOKENIZERS[@]}" >&2
      exit 1
    fi
    echo "[debug] SIGTERM detokenizer ${DETOKENIZERS[0]}"
    kill -TERM "${DETOKENIZERS[0]}" ;;
  clean) echo "[debug] SIGTERM server $SERVER"; kill -TERM "$SERVER" ;;
  group) echo "[debug] SIGTERM process group $SERVER"; kill -TERM -- "-$SERVER" ;;
esac
# Diagnostics include a 5-second settle and a 60-second CUDA-dump wait, plus stack collection.
for ((i=0; i<${STOP_WAIT_S:-180}; i++)); do
  kill -0 "$SERVER" 2>/dev/null || break
  sleep 1
done
if kill -0 "$SERVER" 2>/dev/null; then
  tail -n 60 "$LOG_DIR/sglang.log" >&2
  echo "[debug] server did not stop before the deadline; cleaning up owned processes" >&2
  exit 1
fi
RC=0
wait "$SERVER" || RC=$?
echo "[debug] server exit code=$RC"
echo "[debug] sglang log tail:"
tail -n 60 "$LOG_DIR/sglang.log" | cut -c1-250
cleanup
SERVER=
echo "[debug] left at exit:"
ps -eo pid,ppid,pgid,stat,comm | cut -c1-120
echo "[debug] done, exiting 0"
