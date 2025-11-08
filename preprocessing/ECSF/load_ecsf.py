# load_ecsf.py
import csv
import ast
import os
from cassandra.cluster import Cluster
from cassandra.query import PreparedStatement
from cassandra import InvalidRequest

DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # Same directory as this script
CONTACT_POINTS = ["127.0.0.1"]  # change if your Cassandra is remote
KEYSPACE = "ecsf"

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def to_list_from_str(s):
    # CSV may store python-style lists; handle '', '[]', or JSON-like strings.
    if not s or s.strip() == "":
        return []
    # Try safe eval for tuples/lists saved as Python repr
    try:
        val = ast.literal_eval(s)
        return val
    except Exception:
        # fallback: simple split on semicolon or comma
        return [x.strip() for x in s.split(",") if x.strip()]

def main():
    # connect
    cluster = Cluster(CONTACT_POINTS)
    session = cluster.connect()
    # create keyspace if not exists
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
        WITH replication = {{'class':'SimpleStrategy','replication_factor':1}}
    """)
    session.set_keyspace(KEYSPACE)

    # Prepare statements
    insert_work_role = session.prepare("""
        INSERT INTO work_role_by_id (work_role_id, title, alt_titles, summary_statement, mission, tks_ids, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """)
    insert_role_with_tks = session.prepare("""
        INSERT INTO role_with_tks (work_role_id, title, alt_titles, summary_statement, mission, tks)
        VALUES (?, ?, ?, ?, ?, ?)
    """)
    insert_roles_by_title = session.prepare("""
        INSERT INTO roles_by_title (title_key, work_role_id, is_alt, canonical_title) VALUES (?, ?, ?, ?)
    """)
    insert_roles_by_tks = session.prepare("""
        INSERT INTO roles_by_tks (tks_id, work_role_id) VALUES (?, ?)
    """)
    insert_tks_by_id = session.prepare("""
        INSERT INTO tks_by_id (tks_id, type, description) VALUES (?, ?, ?)
    """)

    # Load and insert tks_by_id
    tks_path = os.path.join(DATA_DIR, "tks_by_id.csv")
    if os.path.exists(tks_path):
        for r in read_csv(tks_path):
            tks_id = r.get("tks_id") or r.get("id") or r.get("tks_id")
            typ = r.get("type")
            desc = r.get("description")
            session.execute(insert_tks_by_id, (tks_id, typ, desc))

    # Load and insert work_role_by_id
    roles_path = os.path.join(DATA_DIR, "work_role_by_id.csv")
    if os.path.exists(roles_path):
        for r in read_csv(roles_path):
            # tks_ids might be stored as "['K0001','S0001']" or as CSV "K0001,S0001"
            tks_ids = to_list_from_str(r.get("tks_ids", "[]"))
            # metadata column could be empty or a dict repr; we keep minimal
            metadata = {}
            if r.get("ingested_at"):
                metadata["ingested_at"] = r.get("ingested_at")
            # alt_titles could be string repr of list
            alt_titles = to_list_from_str(r.get("alt_titles", "[]"))
            try:
                session.execute(insert_work_role, (int(r["work_role_id"]), r.get("title"), alt_titles,
                                                   r.get("summary_statement"), r.get("mission"), tks_ids, metadata))
            except InvalidRequest as e:
                print("Failed to insert work_role:", e, r)

    # Load and insert role_with_tks (tks likely a list of tuples repr)
    role_with_tks_path = os.path.join(DATA_DIR, "role_with_tks.csv")
    if os.path.exists(role_with_tks_path):
        for r in read_csv(role_with_tks_path):
            tks = to_list_from_str(r.get("tks", "[]"))
            # ensure tuples are converted to tuples (ast.literal_eval handles)
            try:
                session.execute(insert_role_with_tks, (int(r["work_role_id"]), r.get("title"),
                                                       to_list_from_str(r.get("alt_titles", "[]")),
                                                       r.get("summary_statement"), r.get("mission"), tks))
            except Exception as e:
                print("role_with_tks insert err:", e)

    # Insert roles_by_title
    roles_by_title_path = os.path.join(DATA_DIR, "roles_by_title.csv")
    if os.path.exists(roles_by_title_path):
        for r in read_csv(roles_by_title_path):
            title_key = r.get("title_key")
            work_role_id = int(r.get("work_role_id"))
            is_alt = (r.get("is_alt", "False").lower() in ("true", "1", "t"))
            canonical_title = r.get("canonical_title")
            session.execute(insert_roles_by_title, (title_key, work_role_id, is_alt, canonical_title))

    # Insert roles_by_tks
    roles_by_tks_path = os.path.join(DATA_DIR, "roles_by_tks.csv")
    if os.path.exists(roles_by_tks_path):
        for r in read_csv(roles_by_tks_path):
            tks_id = r.get("tks_id")
            work_role_id = int(r.get("work_role_id"))
            session.execute(insert_roles_by_tks, (tks_id, work_role_id))

    print("Data load finished. Verify with cqlsh or sample selects.")
    cluster.shutdown()

if __name__ == "__main__":
    main()
