#!/usr/bin/env bash
# Replay the reduced media against a running voice-timeline service.
# Usage: BASE_URL=http://127.0.0.1:8093 ./replay.sh
set -euo pipefail
: "${BASE_URL:?set BASE_URL to a running voice-timeline service}"
cd "$(dirname "$0")"
SOURCE_ID="replay_data_split_by_fmt_order"
PAYLOAD=$(python3 -I -c "import base64;print(base64.b64encode(open('reduced.wav','rb').read()).decode())")
curl -sS -X PUT "$BASE_URL/v1/sources/$SOURCE_ID" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"replay\",\"wav_base64\":\"$PAYLOAD\"}" | python3 -m json.tool
echo
curl -sS "$BASE_URL/v1/sources/$SOURCE_ID" | python3 -m json.tool
