import sqlite3

DB_PATH = "posted.db"

schema = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY,
  link_hash TEXT UNIQUE,
  link TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(schema)
    conn.commit()
    conn.close()
    print(f"Base de datos creada/verificada en: {DB_PATH}")

if __name__ == "__main__":
    main()
