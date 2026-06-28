#!/usr/bin/env bash
# =============================================================================
# Tahap 1: Inisialisasi replica set (configsvr, shard1, shard2)
# Dijalankan SEBELUM mongos, agar configReplSet sudah punya PRIMARY
# saat mongo-router pertama kali menyambung.
# =============================================================================
set -euo pipefail

CONFIGSVR="mongo-configsvr:27017"
SHARD1="mongo-shard1:27017"
SHARD2="mongo-shard2:27017"

wait_for() {
  local host="$1"
  echo "  -> Menunggu ${host} ..."
  until mongosh --quiet --host "${host}" --eval "db.adminCommand('ping')" >/dev/null 2>&1; do
    sleep 2
  done
  echo "  -> ${host} siap."
}

init_rs() {
  local host="$1"; local rs="$2"
  echo "[*] Init replSet ${rs} pada ${host} ..."
  mongosh --quiet --host "${host}" --eval "
    try {
      rs.initiate({ _id: '${rs}', members: [{ _id: 0, host: '${host}' }] });
      print('    ${rs} diinisialisasi.');
    } catch(e) {
      if (/already initialized/i.test(e.message)) {
        print('    ${rs} sudah ada, skip.');
      } else { throw e; }
    }
  "
}

wait_for_primary() {
  local host="$1"; local rs="$2"
  echo "  -> Menunggu PRIMARY di ${rs} ..."
  until mongosh --quiet --host "${host}" --eval "db.hello().isWritablePrimary" 2>/dev/null | grep -q true; do
    sleep 2
  done
  echo "  -> PRIMARY siap: ${host}"
}

echo "========================================================"
echo " TAHAP 1: Inisialisasi Replica Set"
echo "========================================================"

wait_for "${CONFIGSVR}"
wait_for "${SHARD1}"
wait_for "${SHARD2}"

init_rs "${CONFIGSVR}" "configReplSet"
init_rs "${SHARD1}"    "shard1ReplSet"
init_rs "${SHARD2}"    "shard2ReplSet"

wait_for_primary "${CONFIGSVR}" "configReplSet"
wait_for_primary "${SHARD1}"    "shard1ReplSet"
wait_for_primary "${SHARD2}"    "shard2ReplSet"

echo "========================================================"
echo " TAHAP 1 SELESAI — semua replica set punya PRIMARY"
echo "========================================================"
