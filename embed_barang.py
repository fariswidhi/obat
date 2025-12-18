import os
import json
import argparse
import sys
import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer


def get_conn(args):
    try:
        return mysql.connector.connect(
            host=args.host or os.getenv("APOTEK_DB_HOST", "localhost"),
            user=args.user or os.getenv("APOTEK_DB_USER", "root"),
            password=args.password or os.getenv("APOTEK_DB_PASSWORD", "root"),
            database=args.database or os.getenv("APOTEK_DB_NAME", "apotek"),
            port=int(args.port or os.getenv("APOTEK_DB_PORT", "3306")),
        )
    except mysql.connector.Error as e:
        print("Gagal konek ke database. Gunakan argumen --host --user --password --database --port atau set env APOTEK_DB_*.")
        print(f"Detail: {e}")
        sys.exit(1)


def get_column_type(cur, table, col):
    cur.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (col,))
    row = cur.fetchone()
    if not row:
        return None
    return row[1]


def ensure_storage(cur, table, id_col):
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN embedding JSON")
        return {"mode": "column", "type": "json"}
    except mysql.connector.Error:
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN embedding TEXT")
            return {"mode": "column", "type": "text"}
        except mysql.connector.Error:
            id_type = get_column_type(cur, table, id_col) or "INT"
            try:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {table}_embedding ("
                    f"{id_col} {id_type} NOT NULL, "
                    f"embedding JSON, "
                    f"PRIMARY KEY ({id_col}), "
                    f"FOREIGN KEY ({id_col}) REFERENCES {table}({id_col}) ON DELETE CASCADE)"
                )
                return {"mode": "table", "type": "json", "name": f"{table}_embedding"}
            except mysql.connector.Error:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {table}_embedding ("
                    f"{id_col} {id_type} NOT NULL, "
                    f"embedding TEXT, "
                    f"PRIMARY KEY ({id_col}), "
                    f"FOREIGN KEY ({id_col}) REFERENCES {table}({id_col}) ON DELETE CASCADE)"
                )
                return {"mode": "table", "type": "text", "name": f"{table}_embedding"}


def get_primary_key(cur, table):
    cur.execute(f"SHOW KEYS FROM {table} WHERE Key_name = 'PRIMARY'")
    rows = cur.fetchall()
    if rows:
        return rows[0][4]
    cur.execute(f"SHOW COLUMNS FROM {table}")
    cols = [r[0] for r in cur.fetchall()]
    if "id" in cols:
        return "id"
    return cols[0]


def get_text_columns(cur, table):
    candidates = ["nama", "nama_barang", "deskripsi", "keterangan", "aturan_pakai", "komposisi"]
    cur.execute(f"SHOW COLUMNS FROM {table}")
    cols = [r[0] for r in cur.fetchall()]
    return [c for c in candidates if c in cols]


def build_text(row, text_cols):
    parts = []
    for c in text_cols:
        v = row.get(c)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip()


def fetch_batch(cur, table, id_col, text_cols, limit, offset):
    select_cols = ", ".join([id_col] + text_cols)
    cur.execute(f"SELECT {select_cols} FROM {table} LIMIT %s OFFSET %s", (limit, offset))
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    out = []
    for r in rows:
        out.append({col_names[i]: r[i] for i in range(len(col_names))})
    return out


def update_embedding(cur, storage, table, id_col, _id, vec_json):
    if storage["mode"] == "column":
        cur.execute(f"UPDATE {table} SET embedding = %s WHERE {id_col} = %s", (vec_json, _id))
    else:
        side = storage["name"]
        cur.execute(
            f"INSERT INTO {side} ({id_col}, embedding) VALUES (%s, %s) "
            f"ON DUPLICATE KEY UPDATE embedding = VALUES(embedding)",
            (_id, vec_json),
        )


def count_rows(cur, table):
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def run(args):
    conn = get_conn(args)
    conn.autocommit = True
    cur = conn.cursor()
    table = args.table
    id_col = get_primary_key(cur, table)
    text_cols = get_text_columns(cur, table)
    if not text_cols:
        raise RuntimeError("Tidak ada kolom teks yang cocok di tabel.")
    storage = ensure_storage(cur, table, id_col)
    model = SentenceTransformer(args.model)
    total = count_rows(cur, table)
    offset = 0
    while offset < total:
        batch = fetch_batch(cur, table, id_col, text_cols, args.batch_size, offset)
        offset += args.batch_size
        texts = []
        ids = []
        for row in batch:
            t = build_text(row, text_cols)
            if t:
                texts.append(t)
                ids.append(row[id_col])
        if not texts:
            continue
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        for _id, v in zip(ids, vecs):
            vec_json = json.dumps(np.asarray(v, dtype=np.float32).tolist())
            update_embedding(cur, storage, table, id_col, _id, vec_json)
        print(f"Processed {min(offset, total)}/{total}")
    cur.close()
    conn.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--database", default=None)
    p.add_argument("--port", default=None)
    p.add_argument("--table", default="barang")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
