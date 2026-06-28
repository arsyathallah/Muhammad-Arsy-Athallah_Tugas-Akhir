// =============================================================================
// Pendaftaran shard & sharding collection (dijalankan di mongos)
//
// Konfigurasi HARUS sinkron dengan src/config.py:
//   - MONGODB_DATABASE   = "recommendation_system"
//   - MONGODB_SHARD_KEYS = hashed shard key per collection
//
// Idempoten: error "already a shard"/"already sharded" diabaikan.
// =============================================================================

const DB_NAME = "recommendation_system";

// Daftar shard: <replSetName>/<host:port internal>
const SHARDS = [
  "shard1ReplSet/mongo-shard1:27017",
  "shard2ReplSet/mongo-shard2:27017",
];

// Sinkron dengan MONGODB_SHARD_KEYS di src/config.py
const SHARD_KEYS = {
  reviews:          { user_id: "hashed" },
  ratings:          { user_id: "hashed" },
  recommendations:  { user_id: "hashed" },
  product_profiles: { product_id: "hashed" },
  user_profiles:    { user_id: "hashed" },
};

function safe(label, fn) {
  try {
    fn();
    print("  [OK]   " + label);
  } catch (e) {
    if (/already a shard|already sharded|already exists/i.test(e.message)) {
      print("  [SKIP] " + label + " (sudah ada)");
    } else {
      print("  [ERR]  " + label + " -> " + e.message);
      throw e;
    }
  }
}

print("[*] Mendaftarkan shard ...");
SHARDS.forEach(function (s) {
  safe("addShard " + s, function () {
    sh.addShard(s);
  });
});

print("[*] Mengaktifkan sharding pada database '" + DB_NAME + "' ...");
safe("enableSharding " + DB_NAME, function () {
  sh.enableSharding(DB_NAME);
});

print("[*] Sharding collection (hashed key) ...");
Object.keys(SHARD_KEYS).forEach(function (coll) {
  const ns = DB_NAME + "." + coll;
  const key = SHARD_KEYS[coll];
  safe("shardCollection " + ns + " -> " + JSON.stringify(key), function () {
    // Index hashed dibuat otomatis oleh shardCollection bila belum ada
    sh.shardCollection(ns, key);
  });
});

print("[*] Ringkasan shard:");
db.adminCommand({ listShards: 1 }).shards.forEach(function (s) {
  print("    - " + s._id + "  ->  " + s.host);
});

print("[DONE] Inisialisasi sharding selesai.");
