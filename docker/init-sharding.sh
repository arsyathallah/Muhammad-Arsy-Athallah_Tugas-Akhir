#!/usr/bin/env bash
# =============================================================================
# Tahap 2: Daftarkan shard ke mongos & shard collection
# Dijalankan SETELAH mongos (mongo-router) sudah healthy.
# =============================================================================
set -euo pipefail

ROUTER="mongo-router:27017"

wait_for() {
  local host="$1"
  echo "  -> Menunggu ${host} ..."
  until mongosh --quiet --host "${host}" --eval "db.adminCommand('ping')" >/dev/null 2>&1; do
    sleep 2
  done
  echo "  -> ${host} siap."
}

echo "========================================================"
echo " TAHAP 2: Mendaftarkan shard & sharding collection"
echo "========================================================"

wait_for "${ROUTER}"

echo "[*] Menjalankan init-mongo-sharding.js ..."
mongosh --quiet --host "${ROUTER}" /scripts/init-mongo-sharding.js

echo "========================================================"
echo " TAHAP 2 SELESAI — cluster siap digunakan"
echo "========================================================"
