import sqlite3
import hashlib
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

DB_PATH = "posted.db"

# ---------- Normalización de URL ----------
def normalize_url(u: str) -> str:
    """Limpia parámetros de tracking y normaliza para comparar URLs."""
    if not u:
        return u
    s = urlsplit(u)
    netloc = s.netloc.lower()
    path = s.path or "/"

    # Parámetros a eliminar
    disallow = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
        "gclid", "fbclid", "mc_cid", "mc_eid"
    }
    q = [(k, v) for k, v in parse_qsl(s.query, keep_blank_values=True) if k not in disallow]
    query = urlencode(q)

    return urlunsplit((s.scheme, netloc, path, query, ""))

# ---------- Inicialización ----------
def init_db():
    """Crea la base de datos y tablas si no existen."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts(
            id INTEGER PRIMARY KEY,
            link_hash TEXT UNIQUE,
            link TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state(
            k TEXT PRIMARY KEY,
            v TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

# ---------- Estado ----------
def get_state(conn, key, default=None):
    cur = conn.execute("SELECT v FROM state WHERE k=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default

def set_state(conn, key, value):
    conn.execute("""
        INSERT INTO state(k,v) VALUES(?,?)
        ON CONFLICT(k) DO UPDATE SET v=excluded.v,
        updated_at=CURRENT_TIMESTAMP
    """, (key, value))
    conn.commit()

# ---------- Posts publicados ----------
def _hash_link(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def already_posted(conn, link):
    """Devuelve True si el link o su versión normalizada ya están en la BD."""
    norm = normalize_url(link or "")
    h_norm = _hash_link(norm)
    h_raw = _hash_link(link or "")
    cur = conn.execute(
        "SELECT 1 FROM posts WHERE link_hash IN (?,?)",
        (h_norm, h_raw)
    )
    return cur.fetchone() is not None

def mark_posted(conn, link):
    """Guarda un link como publicado."""
    norm = normalize_url(link or "")
    h_norm = _hash_link(norm)
    conn.execute(
        "INSERT OR IGNORE INTO posts (link_hash, link) VALUES (?,?)",
        (h_norm, norm)
    )
    conn.commit()
