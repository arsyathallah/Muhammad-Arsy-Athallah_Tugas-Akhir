#!/usr/bin/env bash
# =============================================================================
# Inisialisasi MongoDB Sharded Cluster
#
# Urutan:
#   1. Inisialisasi replica set config server (configReplSet)
#   2. Inisialisasi replica set shard1 (shard1ReplSet)
#   3. Inisialisasi replica set shard2 (shard2ReplSet)
#   4. Daftarkan shard ke mongos & shard collection (init-mongo-sharding.js)
#
# Idempoten: aman dijalankan ulang. "already initialized" diabaikan.
# =============================================================================
set -euo pipefail

CONFIGSVR="mongo-configsvr:27017"
SHARD1="mongo-shard1:27017"
SHARD2="mongo-shard2:27017"
ROUTER="mongo-router:27017"

# Tunggu sampai sebuah mongod/mongos siap menerima koneksi
wait_for() {
  local host="$1"
  echo "  -> Menunggu ${host} ..."
  until mongosh --quiet --host "${host}" --eval "db.adminCommand('ping')" >/dev/null 2>&1; do
    sleep 2
  done
  echo "  -> ${host} siap."
}

# Inisialisasi replica set bila belum terinisialisasi
init_rs() {
  local host="$1"; local rs="$2"
  echo "[*] Inisialisasi replica set ${rs} pada ${host}"
  mongosh --quiet --host "${host}" --eval "
    try {
      rs.initiate({
        _id: '${rs}',
        members: [{ _id: 0, host: '${host}' }]
      });
      print('    ${rs} diinisialisasi.');
    } catch (e) {
      if (e.codeName === 'AlreadyInitialized' || /already initialized/i.test(e.message)) {
        print('    ${rs} sudah terinisialisasi sebelumnya.');
      } else {
        throw e;
      }
    }
  "
}

echo "============================================================"
echo " SETUP MONGODB SHARDED CLUSTER"
echo "============================================================"

wait_for "${CONFIGSVR}"
wait_for "${SHARD1}"
wait_for "${SHARD2}"

init_rs "${CONFIGSVR}" "configReplSet"
init_rs "${SHARD1}"    "shard1ReplSet"
init_rs "${SHARD2}"    "shard2ReplSet"

# Beri waktu tiap replica set memilih PRIMARY
echo "[*] Menunggu PRIMARY terpilih di tiap replica set ..."
for host in "${CONFIGSVR}" "${SHARD1}" "${SHARD2}"; do
  until mongosh --quiet --host "${host}" --eval "db.hello().isWritablePrimary" 2>/dev/null | grep -q true; do
    sleep 2
  done
  echo "  -> PRIMARY siap di ${host}"
done

wait_for "${ROUTER}"

echo "[*] Mendaftarkan shard & sharding collection via mongos ..."
mongosh --quiet --host "${ROUTER}" /scripts/init-mongo-sharding.js

echo "============================================================"
echo " CLUSTER SIAP. Cek dengan: sh.status()"
echo "============================================================"
