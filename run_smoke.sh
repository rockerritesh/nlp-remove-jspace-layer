#!/bin/bash
# Runs entirely on the VM, detached. Restarts the server with current server.py,
# waits for readiness, runs one baseline-vs-ablated generation, writes gen.json.
# Progress/markers go to smoke_result.txt so the (flaky) client only needs short
# SSH polls of that file.
cd ~/layer-ablation || exit 1
exec > smoke_result.txt 2>&1

echo ">>> RESTART $(date +%T)"
pkill -f "load-4bit --host" 2>/dev/null
sleep 2
rm -f server.log gen.json
setsid --fork bash -c "exec ~/.venv/bin/python server.py --load-4bit --host 127.0.0.1 --port 8000 > server.log 2>&1" </dev/null >/dev/null 2>&1

for i in $(seq 1 40); do
  H=$(curl -s -m 4 http://127.0.0.1:8000/health 2>/dev/null)
  if echo "$H" | grep -q '"status":"ok"'; then echo ">>> READY $(date +%T)"; break; fi
  if ! ps aux | grep -q "[l]oad-4bit"; then echo ">>> DIED"; tail -20 server.log; echo ">>> DONE"; exit 0; fi
  echo "[$i] loading..."
  sleep 6
done

echo ">>> GENERATING $(date +%T)"
curl -s -m 300 -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"In one sentence, why is the sky blue?","layer":14,"max_new_tokens":80,"temperature":0}' \
  > gen.json 2>/dev/null

if ~/.venv/bin/python -c "import json,sys; d=json.load(open('gen.json')); print('OK' if 'baseline' in d else 'BAD')" 2>/dev/null | grep -q OK; then
  echo ">>> GENERATE OK $(date +%T)"
else
  echo ">>> GENERATE FAIL"; head -c 600 gen.json
fi
echo ">>> DONE"
