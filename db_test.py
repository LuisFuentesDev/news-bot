import sqlite3, hashlib

DB_PATH = "posted.db"
test_link = "https://ejemplo.com/noticia-123"
h = hashlib.sha256(test_link.encode("utf-8")).hexdigest()

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO posts (link_hash, link) VALUES (?,?)", (h, test_link))
    conn.commit()
    cur.execute("SELECT id, link FROM posts WHERE link_hash=?", (h,))
    print("Fila:", cur.fetchone())
