
import os
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path("C:/data")

def _find_bin(name: str) -> str:
    if sys.platform != "win32":
        return name
    for ver in ["8.0", "7.0", "6.0", "5.0"]:
        p = Path(f"C:/Program Files/MongoDB/Server/{ver}/bin/{name}.exe")
        if p.exists():
            return str(p)
    return name

MONGOD  = _find_bin("mongod")
MONGOSH = _find_bin("mongosh")
NODES = [
    {"port": 27017, "dbpath": DATA_ROOT / "rs0-0"},
    {"port": 27018, "dbpath": DATA_ROOT / "rs0-1"},
    {"port": 27019, "dbpath": DATA_ROOT / "rs0-2"},
]
REPLICA_SET = "rs0"

def create_dirs():
    for node in NODES:
        node["dbpath"].mkdir(parents=True, exist_ok=True)
        print(f"  Dir: {node['dbpath']}")

def start_nodes():
    procs = []
    for node in NODES:
        cmd = [
            MONGOD,
            "--port",    str(node["port"]),
            "--dbpath",  str(node["dbpath"]),
            "--replSet", REPLICA_SET,
            "--bind_ip", "localhost",
            "--logpath", str(node["dbpath"] / "mongod.log"),
            "--fork",
        ]
        if sys.platform == "win32":
            cmd = [c for c in cmd if c != "--fork"]
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                close_fds=True,
            )
        else:
            subprocess.run(cmd, check=True)
        print(f"  mongod port {node['port']} started")
        procs.append(node["port"])
    return procs

def init_replica_set():
    time.sleep(5)
    try:
        from pymongo import MongoClient
        from pymongo.errors import OperationFailure
    except ImportError:
        print("  ERROR: pymongo tidak terinstall — pip install pymongo")
        sys.exit(1)

    client = MongoClient("localhost", 27017, serverSelectionTimeoutMS=10000)
    config = {
        "_id": REPLICA_SET,
        "members": [
            {"_id": 0, "host": "localhost:27017"},
            {"_id": 1, "host": "localhost:27018"},
            {"_id": 2, "host": "localhost:27019"},
        ],
    }
    try:
        result = client.admin.command("replSetInitiate", config)
        print(f"  Replica set diinisialisasi   (ok={result.get('ok')})")
    except OperationFailure as e:
        if "already initialized" in str(e):
            print("  Replica set sudah terinisialisasi sebelumnya ")
        else:
            print(f"  ERROR: {e}")
            sys.exit(1)
    finally:
        client.close()

def check_status():
    time.sleep(5)
    try:
        from pymongo import MongoClient
        client = MongoClient(
            f"mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet={REPLICA_SET}",
            serverSelectionTimeoutMS=15000,
        )
        status = client.admin.command("replSetGetStatus")
        print("\n  Status replica set:")
        for m in status.get("members", []):
            print(f"    {m['name']}  →  {m['stateStr']}")
        client.close()
    except Exception as e:
        print(f"  Tidak bisa ambil status: {e}")

def main():
    print("=" * 55)
    print("Setup MongoDB Replica Set (rs0) — 3 nodes")
    print("=" * 55)

    print("\n[1] Membuat direktori data...")
    create_dirs()

    print("\n[2] Menjalankan mongod instances...")
    start_nodes()

    print("\n[3] Inisialisasi replica set...")
    init_replica_set()

    print("\n[4] Cek status...")
    check_status()

    print("\n MongoDB Replica Set aktif!")
    print("   Primary : localhost:27017")
    print("   Secondary: localhost:27018")
    print("   Secondary: localhost:27019")
    print("\nLanjut jalankan:")
    print("   python streamlit_app/scripts/migrate_to_mongo.py")

if __name__ == "__main__":
    main()
