# automation_news.py — rotación por categoría + limpieza autor + sin imágenes en cuerpo + tweet v2 con límite mensual
import os, re, io, time, hashlib, sqlite3, datetime as dt
from urllib.parse import urljoin
import requests, feedparser
from bs4 import BeautifulSoup
from readability import Document
from PIL import Image
from dotenv import load_dotenv
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import tweepy
import shutil
import unicodedata

load_dotenv()

WP_URL = os.getenv("WP_URL", "").rstrip("/")
WP_USER = os.getenv("WP_USER")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")
POST_MODE = os.getenv("POST_MODE", "summary")  # summary | full_html
LANG = os.getenv("LANG", "es")

# Tweets
TW_ENABLED = os.getenv("TW_ENABLED", "1")                 # "1" para habilitar, "0" para desactivar sin tocar código
TW_MONTHLY_LIMIT = int(os.getenv("TW_MONTHLY_LIMIT", "500"))  # Free plan: 500 writes/mes

assert WP_URL and WP_USER and WP_APP_PASSWORD, "Faltan WP_URL/WP_USER/WP_APP_PASSWORD en .env"

# ---- ORDEN DE ROTACIÓN ----
CATEGORY_ORDER = ["REGIONAL", "NACIONAL", "INTERNACIONAL", "DEPORTES"]

# ---- FEEDS por categoría ----
FEEDS_BY_CATEGORY = {
    "REGIONAL": [
        "https://www.soychile.cl/rss",
        "https://rss.app/feeds/CYNJPggNEkFxnne0.xml",
        "https://rss.app/feeds/MAgTVCRXwboo1wpZ.xml",
    ],
    "NACIONAL": [
        "https://www.emol.com/rss/rss.asp",
        "https://rss.app/feeds/QuPZil06i75x4J0b.xml",
        "https://rss.app/feeds/KXD7V1vwEsU5FFoc.xml",
    ],
    "INTERNACIONAL": [
        "https://cnnespanol.cnn.com/feed/",
        "https://www.infobae.com/america/feed",
        "https://rss.app/feeds/tipE19EolhTiyutE.xml",
        "https://rss.app/feeds/gWh6Ibrm0CzSX2qZ.xml",
    ],
    "DEPORTES": [
        "https://www.ole.com.ar/rss/ultimas-noticias/",
        "https://www.marca.com/rss/portada.xml",
        "https://rss.app/feeds/24FkKxNQEXFjnS3c.xml",
        "https://rss.app/feeds/mrcUDGUfBSKUZVCA.xml",
    ],
}

DB_PATH = "posted.db"
HEADERS = {"User-Agent": "NewsBot/1.0 (+winforma.cl)", "Accept": "text/html,application/xhtml+xml"}

API_BASES = [f"{WP_URL}/wp-json/wp/v2", f"{WP_URL}/?rest_route=/wp/v2"]
HEADERS_JSON = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "NewsBot/1.0"}
HEADERS_BIN = {"Accept": "application/json", "User-Agent": "NewsBot/1.0"}
AUTH = (WP_USER, WP_APP_PASSWORD)

def make_seo_title(title: str, max_len=60) -> str:
    t = " ".join(title.split())
    return (t[:max_len-1] + "…") if len(t) > max_len else t

def make_meta_description(paragraphs, max_len=160) -> str:
    base = " ".join(paragraphs[:2]) if paragraphs else ""
    base = " ".join(base.split())
    return (base[:max_len-1] + "…") if len(base) > max_len else base

def slugify(text: str) -> str:
    t = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('ascii')
    t = re.sub(r'[^a-z0-9\- ]+', '', t.lower()).strip()
    t = re.sub(r'\s+', '-', t)
    t = re.sub(r'-{2,}', '-', t)
    return t[:90] or "nota"

HASHTAGS = {
    "REGIONAL": ["#Chile", "#Regional"],
    "NACIONAL": ["#Chile", "#Nacional"],
    "INTERNACIONAL": ["#Internacional"],
    "DEPORTES": ["#Deportes"],
}

def add_utm(u: str, **utm) -> str:
    s = urlsplit(u)
    q = dict(parse_qsl(s.query, keep_blank_values=True))
    q.update(utm)
    return urlunsplit((s.scheme, s.netloc, s.path, urlencode(q), s.fragment))

def build_tweet_text(title: str, url: str, cat: str) -> str:
    # UTM solo para X (no afecta dedupe; marcas con la canonical final)
    url_utm = add_utm(url, utm_source="x", utm_medium="social", utm_campaign="newsbot")
    tags = HASHTAGS.get(cat, [])
    base = f"{title}\n{url_utm}"
    if tags:
        tail = " " + " ".join(tags[:2])  # 1–2 hashtags máx
        if len(base) + len(tail) <= 280:
            base += tail
    return base[:280]

# ---------- HTTP POST con redirecciones seguras ----------
def _post_with_redirect(url, *, json=None, data=None, headers=None, max_hops=3):
    current = url
    for _ in range(max_hops + 1):
        r = requests.post(current, json=json, data=data, headers=headers, auth=AUTH,
                          timeout=30, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("Location")
            if not loc:
                r.raise_for_status()
            current = loc
            continue
        return r
    raise RuntimeError("Demasiadas redirecciones en POST")

# ---------- Categorías WP ----------
def wp_get_categories_by_search(name):
    for base in API_BASES:
        url = f"{base}/categories?per_page=100&search={name}"
        r = requests.get(url, headers=HEADERS_JSON, auth=AUTH, timeout=20)
        if r.ok:
            return r.json()
    return []

def wp_create_category(name, parent=None):
    payload = {"name": name}
    if parent: payload["parent"] = parent
    for base in API_BASES:
        r = _post_with_redirect(f"{base}/categories", json=payload, headers=HEADERS_JSON)
        if r.ok:
            return r.json()["id"]
    raise RuntimeError(f"No se pudo crear categoría {name}: {r.status_code} {r.text[:200]}")

def get_or_create_category_id_exact(name):
    cats = wp_get_categories_by_search(name)
    for c in cats:
        if c.get("name","").strip().lower() == name.strip().lower():
            return c["id"]
    return wp_create_category(name)

# ---------- Posts / Media ----------
def wp_create_post(title, html_content, featured_media_id=None, status="draft",
                   category_id=None, excerpt=None, slug=None):
    payload = {"title": title, "content": html_content, "status": status}
    if excerpt: payload["excerpt"] = excerpt
    if slug: payload["slug"] = slug
    if category_id: payload["categories"] = [category_id]
    if featured_media_id: payload["featured_media"] = featured_media_id
    last_r = None
    for base in API_BASES:
        r = _post_with_redirect(f"{base}/posts/", json=payload, headers=HEADERS_JSON)
        last_r = r
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "id" in data:
                return data["id"], data.get("link")
    raise RuntimeError(f"No se pudo crear post. Última respuesta: {last_r.status_code if last_r else '??'}")

def wp_upload_media(jpeg_bytes, filename):
    headers = {**HEADERS_BIN, "Content-Type": "image/jpeg",
               "Content-Disposition": f'attachment; filename="{filename}"'}
    last_r = None
    for base in API_BASES:
        r = _post_with_redirect(f"{base}/media/", data=jpeg_bytes, headers=headers)
        last_r = r
        if r.ok:
            return r.json()["id"]
    raise RuntimeError(f"No se pudo subir media. Última respuesta: {last_r.status_code if last_r else '??'}")

# ---------- DB ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS posts(
        id INTEGER PRIMARY KEY, link_hash TEXT UNIQUE, link TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS state(
        k TEXT PRIMARY KEY, v TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    return conn

def get_state(conn, key, default=None):
    cur = conn.execute("SELECT v FROM state WHERE k=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default

def set_state(conn, key, value):
    conn.execute("""INSERT INTO state(k,v) VALUES(?,?)
                    ON CONFLICT(k) DO UPDATE SET v=excluded.v, updated_at=CURRENT_TIMESTAMP""",
                 (key, value))
    conn.commit()

def _hash_link(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def already_posted(conn, link):
    """
    Devuelve True si el link ya fue publicado.
    - Normaliza la URL antes de hashear.
    - También verifica el hash 'raw' (compatibilidad hacia atrás con filas viejas).
    """
    norm = normalize_url(link or "")
    h_norm = _hash_link(norm)
    h_raw = _hash_link(link or "")
    cur = conn.execute(
        "SELECT 1 FROM posts WHERE link_hash IN (?,?)",
        (h_norm, h_raw)
    )
    return cur.fetchone() is not None

def mark_posted(conn, link):
    """
    Guarda solo la versión normalizada en la tabla (link y hash).
    Así, a futuro todo queda consistente.
    """
    norm = normalize_url(link or "")
    h_norm = _hash_link(norm)
    conn.execute(
        "INSERT OR IGNORE INTO posts (link_hash, link) VALUES (?,?)",
        (h_norm, norm)
    )
    conn.commit()

# ---------- Contador mensual de tweets ----------
def _month_key_utc():
    return dt.datetime.utcnow().strftime("%Y-%m")

def _ensure_month(conn):
    mk = _month_key_utc()
    curr = get_state(conn, "tw_month", "")
    if curr != mk:
        set_state(conn, "tw_month", mk)
        set_state(conn, "tw_writes", "0")

def get_tw_writes(conn):
    _ensure_month(conn)
    v = get_state(conn, "tw_writes", "0") or "0"
    try:
        return int(v)
    except:
        set_state(conn, "tw_writes", "0")
        return 0

def can_tweet(conn):
    _ensure_month(conn)
    count = get_tw_writes(conn)
    return TW_ENABLED == "1" and count < TW_MONTHLY_LIMIT

def record_tweet_success(conn):
    c = get_tw_writes(conn) + 1
    set_state(conn, "tw_writes", str(c))
    return c

# ---------- Limpieza de autor/firmas ----------
BYLINE_PATTERNS = [
    r'^(?:por|by)\s+[^.,|:]{2,80}[:—–-]\s*',
    r'^(?:publicado\s+por|autor(?:a)?):\s+.*?(?:[:|—–-]\s*)',
    r'^(?:redacción|agencias?|efe|afp|ap|reuters|bbc\s+mundo|cnn\s+español)\s+[:—–-]\s*',
    r'^[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ\.]+(?:\s+[A-ZÁÉÍÓÚÑ][\w\.-]+){0,4}\s+—\s*',
]

def strip_byline_prefix(text: str) -> str:
    t = text.lstrip()
    for pat in BYLINE_PATTERNS:
        t2 = re.sub(pat, '', t, flags=re.IGNORECASE)
        if t2 != t:
            t = t2.lstrip()
            break
    head = t[:220]
    if any(k in head.lower() for k in ('publicado', 'autor', 'redacción', 'agencia', 'fecha', 'hora', 'editor')):
        parts = re.split(r'(?:\s[|]\s|—|–|-){1,2}', head, maxsplit=1)
        if len(parts) > 1 and '.' in parts[1]:
            t = parts[1][parts[1].find('.')+1:].lstrip()
    return t

def strip_leading_metadata(text: str) -> str:
    t = text.lstrip()
    t = re.sub(
        r'^(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\s*'
        r'(?:\||—|–|-|,)?\s*(?:publicado.*?|actualizado.*?|a\s+las[:\s]*\d{1,2}:\d{2}.*?|[\w\s:.]+)?\s*',
        '',
        t,
        flags=re.IGNORECASE
    )
    head = t[:260].lower()
    if any(k in head for k in ('publicado', 'actualizado', 'hora', 'redacción', 'agencia', 'editor')):
        first_dot = t.find('.')
        if 0 < first_dot < 220:
            t = t[first_dot+1:].lstrip()
    return t

def strip_author_nodes(soup: BeautifulSoup) -> BeautifulSoup:
    for node in soup.find_all(True, attrs={'class': re.compile(r'(byline|author|meta|credit|source)', re.I)}):
        node.decompose()
    for node in soup.find_all(True, attrs={'id': re.compile(r'(byline|author|meta|credit|source)', re.I)}):
        node.decompose()
    return soup

# ---------- Fetch & parsing ----------
def fetch_url(url, timeout=20):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    raw = r.content
    enc = r.encoding or r.apparent_encoding or "utf-8"
    try:
        html = raw.decode(enc, errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")
    lower = html.lower()
    if "Ã" in html and "charset=iso-8859-1" not in lower and "charset=latin-1" not in lower:
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            try:
                html = html.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                pass
    return html, r.url

def extract_og_image(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("meta", property="og:image")
    if tag and tag.get("content"):
        return urljoin(base_url, tag["content"])
    img = soup.find("img", src=True)
    return urljoin(base_url, img["src"]) if img else None

def to_jpeg_bytes(img_url, target_w=1200, target_h=630):
    r = requests.get(img_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)).convert("RGB")

    # Escala manteniendo proporción y recorta centrado a 1200x630
    scale = max(target_w / im.width, target_h / im.height)
    new = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    left = max(0, (new.width - target_w) // 2)
    top  = max(0, (new.height - target_h) // 2)
    new = new.crop((left, top, left + target_w, top + target_h))

    buf = io.BytesIO()
    new.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
    buf.seek(0)
    return buf.read()

def summarize_html(html, max_words=1000):
    doc = Document(html)
    article_html = doc.summary()
    soup = BeautifulSoup(article_html, "html.parser")
    soup = strip_author_nodes(soup)
    for bad in soup(["script", "style", "aside", "footer", "nav",
                     "figure", "figcaption", "noscript", "img", "picture", "source"]):
        bad.decompose()
    total_words = 0
    clean_paragraphs = []
    for p in soup.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        text = strip_byline_prefix(text)
        text = strip_leading_metadata(text)
        words = text.split()
        if not words:
            continue
        if total_words + len(words) > max_words:
            words = words[:max_words - total_words]
        total_words += len(words)
        clean_paragraphs.append(" ".join(words))
        if total_words >= max_words:
            break
    return clean_paragraphs

def build_post_html(paragraphs):
    return "<div class='winf-body'>" + "".join(
        f"<p style='text-align:justify;'>{p}</p>" for p in paragraphs
    ) + "</div>"

# ---------- Tweets (v2) ----------
def tweet_news(text: str) -> bool:
    if TW_ENABLED != "1":
        print("[X] Tweet desactivado por TW_ENABLED.")
        return False

    ck  = os.getenv("TW_CONSUMER_KEY")
    cs  = os.getenv("TW_CONSUMER_SECRET")
    at  = os.getenv("TW_ACCESS_TOKEN")
    ats = os.getenv("TW_ACCESS_TOKEN_SECRET")
    if not all([ck, cs, at, ats]):
        print("[X] Credenciales de X no configuradas; omito tweet.")
        return False

    try:
        client = tweepy.Client(
            consumer_key=ck,
            consumer_secret=cs,
            access_token=at,
            access_token_secret=ats
        )
        resp = client.create_tweet(text=text[:280])
        ok = bool(getattr(resp, "data", None) and resp.data.get("id"))
        if ok:
            print(f"[X] Tweet enviado: id={resp.data.get('id')}")
            return True
        print("[X ERROR] create_tweet respondió sin id.")
        return False
    except tweepy.TweepyException as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        print(f"[X ERROR] No se pudo publicar en X (status={code}): {e}")
        return False
    except Exception as e:
        print(f"[X ERROR] No se pudo publicar en X: {e}")
        return False

# ---------- Feeds ----------
def pick_entry_for_category(cat_name, max_per_feed=8):
    feeds = FEEDS_BY_CATEGORY.get(cat_name, [])
    for feed in feeds:
        try:
            d = feedparser.parse(feed)
            for entry in d.entries[:max_per_feed]:
                yield entry
        except Exception:
            continue

ACRONYMS = {
    "ONU","OTAN","UE","OMS","OPS","OEA","FMI","BM","BID",
    "PDI","SML","SII","UDI","RN","PS","PC","FA","DC",
    "FBI","CIA","NASA","OEA","G20","APEC","UEFA","FIFA"
}

def normalize_title_case(raw: str) -> str:
    if not raw:
        return raw
    orig = " ".join(raw.split())
    base = orig.lower()
    if base:
        base = base[0].upper() + base[1:]
    tokens = base.split()
    orig_tokens = orig.split()
    fixed = []
    for i, tok in enumerate(tokens):
        pre = "".join(ch for ch in tok if ch in "¿¡(\"'“”‘’[{-")
        suf = "".join(ch for ch in tok if ch in ").,:;!?\"'”’]}-%")
        core = tok.strip("¿¡(\"'“”‘’[{-).,:;!?”’]}-%")
        orig_tok = orig_tokens[i] if i < len(orig_tokens) else tok
        orig_core = orig_tok.strip("¿¡(\"'“”‘’[{-).,:;!?”’]}-%")
        core_up = core.upper()
        if (core_up in ACRONYMS) or (orig_core.isalpha() and 2 <= len(orig_core) <= 5 and orig_core.isupper()):
            new_core = core_up
        else:
            new_core = core
        fixed.append(f"{pre}{new_core}{suf}")
    out = " ".join(fixed)
    for i, ch in enumerate(out):
        if ch.isalpha():
            out = out[:i] + out[i].upper() + out[i+1:]
            break
    return out

def normalize_url(u: str) -> str:
    if not u:
        return u
    s = urlsplit(u)
    netloc = s.netloc.lower()
    path = s.path or "/"
    disallow = {
        "utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id",
        "gclid","fbclid","mc_cid","mc_eid"
    }
    q = [(k, v) for k, v in parse_qsl(s.query, keep_blank_values=True) if k not in disallow]
    query = urlencode(q)
    return urlunsplit((s.scheme, netloc, path, query, ""))

def migrate_posts_table(conn):
    """
    Normaliza y deduplica la tabla posts.
    - Conserva la fila más antigua por cada URL normalizada.
    - Requiere que existan normalize_url() y _hash_link().
    """
    # respaldo del archivo sqlite por si acaso
    try:
        ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        shutil.copyfile(DB_PATH, f"{DB_PATH}.bak.{ts}")
        print(f"[MIGRATE] Backup creado: {DB_PATH}.bak.{ts}")
    except Exception as e:
        print(f"[MIGRATE] No se pudo crear backup (continuo de todos modos): {e}")

    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF;")
    conn.execute("BEGIN;")

    try:
        # Crear tabla nueva con el mismo esquema (y UNIQUE en link_hash)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS posts_new(
                id INTEGER PRIMARY KEY,
                link_hash TEXT UNIQUE,
                link TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Traer todas las filas actuales
        rows = list(cur.execute("SELECT id, link, created_at FROM posts ORDER BY created_at ASC, id ASC;"))

        inserted = 0
        seen_hashes = set()

        for _id, link, created_at in rows:
            norm = normalize_url(link or "")
            h = _hash_link(norm)
            if h in seen_hashes:
                continue  # ya insertamos una fila equivalente (nos quedamos con la más antigua)
            seen_hashes.add(h)
            cur.execute(
                "INSERT OR IGNORE INTO posts_new(link_hash, link, created_at) VALUES (?,?,?);",
                (h, norm, created_at)
            )
            if cur.rowcount > 0:
                inserted += 1

        old_count = cur.execute("SELECT COUNT(*) FROM posts;").fetchone()[0]
        new_count = cur.execute("SELECT COUNT(*) FROM posts_new;").fetchone()[0]

        # Reemplazar tabla
        cur.execute("DROP TABLE posts;")
        cur.execute("ALTER TABLE posts_new RENAME TO posts;")

        conn.commit()
        print(f"[MIGRATE] OK. Antes: {old_count} filas, después (normalizadas y únicas): {new_count}.")
        # Limpieza del archivo
        cur.execute("VACUUM;")
        print("[MIGRATE] VACUUM completado.")
    except Exception as e:
        conn.rollback()
        print(f"[MIGRATE] ERROR, se revirtió la migración: {e}")
        print("[MIGRATE] Puedes restaurar el backup .bak creado si algo quedó mal.")
        return False

    return True

# ---------- Publicación ----------
def publish_one_for_category(conn, category_name, publish_status="publish"):
    cat_id = get_or_create_category_id_exact(category_name)

    for entry in pick_entry_for_category(category_name):
        title = normalize_title_case(entry.get("title") or "(Sin título)")

        # Normaliza la URL del feed
        link = normalize_url(entry.get("link"))
        if not link or already_posted(conn, link):
            continue

        try:
            html, final_url = fetch_url(link)
            # Normaliza la URL final (canonical) después de redirecciones
            final_url_norm = normalize_url(final_url or link)
        except Exception as e:
            print(f"[WARN] No se pudo abrir {link}: {e}")
            continue

        # Imagen destacada (redimensionada 1200x630 para X)
        media_id = None
        try:
            cover = extract_og_image(html, final_url_norm)
            if cover:
                media_id = wp_upload_media(to_jpeg_bytes(cover), "portada.jpg")
        except Exception as e:
            print(f"[WARN] Imagen falló: {e}")

        # --- Genera el contenido del post ---
        if POST_MODE == "full_html":
            soup = BeautifulSoup(Document(html).summary(), "html.parser")
            soup = strip_author_nodes(soup)
            for tag in soup.find_all(["img", "picture", "source", "figure", "figcaption"]):
                tag.decompose()
            content_html = str(soup)
            # Para meta description tomamos los <p> ya limpios
            paragraphs_for_desc = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        else:
            summary = summarize_html(html, max_words=1000)
            content_html = build_post_html(summary)
            paragraphs_for_desc = summary

        # --- SEO: title/description/slug ---
        seo_title = make_seo_title(title)
        meta_desc = make_meta_description(paragraphs_for_desc)
        slug = slugify(title)

        try:
            post_id, post_link = wp_create_post(
                seo_title, content_html,
                featured_media_id=media_id,
                status=publish_status,
                category_id=cat_id,
                excerpt=meta_desc,
                slug=slug
            )

            # Marca como publicada usando la URL final normalizada
            mark_posted(conn, final_url_norm)
            print(f"[OK] {category_name} ({publish_status}): {post_link or post_id}")

            # --- Tweet optimizado (hashtags + UTM) con límite mensual ---
            if publish_status == "publish" and post_link:
                if can_tweet(conn):
                    tweet_text = build_tweet_text(seo_title, post_link, category_name)
                    if tweet_news(tweet_text):
                        used = record_tweet_success(conn)
                        print(f"[X] Writes usados este mes: {used}/{TW_MONTHLY_LIMIT}")
                        time.sleep(5)
                else:
                    used = get_tw_writes(conn)
                    print(f"[X] Límite mensual alcanzado ({used}/{TW_MONTHLY_LIMIT}). Omite tweet hasta reset mensual.")

            return True

        except Exception as e:
            print(f"[ERROR] Falló publicar '{title}' en {category_name}: {e}")
            continue

    print(f"[INFO] No encontré noticia apta para {category_name} en esta pasada.")
    return False

def run_rotating_once(publish_status="publish"):
    conn = init_db()
    try:
        idx = int(get_state(conn, "rotation_idx", "0")) % len(CATEGORY_ORDER)
        cat = CATEGORY_ORDER[idx]
        ok = publish_one_for_category(conn, cat, publish_status=publish_status)
        next_idx = (idx + 1) % len(CATEGORY_ORDER)
        set_state(conn, "rotation_idx", str(next_idx))
        return ok, cat, idx, next_idx
    finally:
        conn.close()

if __name__ == "__main__":
    # Ejecuta migración si se pide por env (una sola vez)
    if os.getenv("MIGRATE_POSTS", "0") == "1":
        conn = init_db()
        try:
            migrate_posts_table(conn)
        finally:
            conn.close()
        raise SystemExit(0)

    ok, cat, idx, nxt = run_rotating_once(publish_status="publish")
    print(f"[ROTATION] idx={idx} cat={cat} -> next={nxt}")
