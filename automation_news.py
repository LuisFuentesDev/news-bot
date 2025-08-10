# automation_news.py — rotación por categoría + limpieza autor + sin imágenes en cuerpo
import os, re, io, time, hashlib, sqlite3
from urllib.parse import urljoin
import requests, feedparser
from bs4 import BeautifulSoup
from readability import Document
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

WP_URL = os.getenv("WP_URL", "").rstrip("/")
WP_USER = os.getenv("WP_USER")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")
POST_MODE = os.getenv("POST_MODE", "summary")  # summary | full_html
LANG = os.getenv("LANG", "es")

assert WP_URL and WP_USER and WP_APP_PASSWORD, "Faltan WP_URL/WP_USER/WP_APP_PASSWORD en .env"

# ---- ORDEN DE ROTACIÓN ----
CATEGORY_ORDER = ["REGIONAL", "NACIONAL", "INTERNACIONAL", "DEPORTES"]

# ---- FEEDS por categoría ----
FEEDS_BY_CATEGORY = {
    "REGIONAL": [
        "https://rss.app/feeds/MAgTVCRXwboo1wpZ.xml",
        "https://rss.app/feeds/CYNJPggNEkFxnne0.xml"
    ],
    "NACIONAL": [
        "https://rss.app/feeds/KXD7V1vwEsU5FFoc.xml",
        "https://rss.app/feed/QuPZil06i75x4J0b",
    ],
    "INTERNACIONAL": [
        "https://rss.app/feeds/gWh6Ibrm0CzSX2qZ.xml",
        "https://rss.app/feeds/tipE19EolhTiyutE.xml",
    ],
    "DEPORTES": [
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
def wp_create_post(title, html_content, featured_media_id=None, status="draft", category_id=None):
    payload = {"title": title, "content": html_content, "status": status}
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

def already_posted(conn, link):
    h = hashlib.sha256(link.encode("utf-8")).hexdigest()
    cur = conn.execute("SELECT 1 FROM posts WHERE link_hash=?", (h,))
    return cur.fetchone() is not None

def mark_posted(conn, link):
    h = hashlib.sha256(link.encode("utf-8")).hexdigest()
    conn.execute("INSERT OR IGNORE INTO posts (link_hash, link) VALUES (?,?)", (h, link))
    conn.commit()

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
    # corta encabezados tipo "Publicado | Fecha ..." si aparecen antes del primer punto
    head = t[:220]
    if any(k in head.lower() for k in ('publicado', 'autor', 'redacción', 'agencia', 'fecha', 'hora', 'editor')):
        parts = re.split(r'(?:\s[|]\s|—|–|-){1,2}', head, maxsplit=1)
        if len(parts) > 1 and '.' in parts[1]:
            t = parts[1][parts[1].find('.')+1:].lstrip()
    return t

def strip_leading_metadata(text: str) -> str:
    t = text.lstrip()

    # Día + fecha [+ separador + publicado/actualizado/hora...]
    t = re.sub(
        r'^(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\s*'
        r'(?:\||—|–|-|,)?\s*(?:publicado.*?|actualizado.*?|a\s+las[:\s]*\d{1,2}:\d{2}.*?|[\w\s:.]+)?\s*',
        '',
        t,
        flags=re.IGNORECASE
    )

    # Si aún quedan metadatos breves antes del primer punto, córtalos
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

def to_jpeg_bytes(img_url):
    r = requests.get(img_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return buf.read()

def summarize_html(html, max_words=1000):
    doc = Document(html)
    article_html = doc.summary()
    soup = BeautifulSoup(article_html, "html.parser")

    # Elimina autores, imágenes y basura
    soup = strip_author_nodes(soup)
    for bad in soup(["script", "style", "aside", "footer", "nav", 
                     "figure", "figcaption", "noscript", "img", "picture", "source"]):
        bad.decompose()

    # Recorre párrafos y limpia texto
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
    """
    Convierte el titular a 'oración' (Primera mayúscula y resto minúsculas),
    preservando siglas. También vuelve a MAYÚSCULAS las siglas conocidas aunque
    vengan en minúsculas. Detecta si el original venía TODO EN MAYÚSCULAS.
    """
    if not raw:
        return raw

    # Normaliza espacios
    orig = " ".join(raw.split())
    base = orig.lower()                      # todo en minúsculas
    if base:
        base = base[0].upper() + base[1:]    # primera letra en mayúscula

    tokens = base.split()
    orig_tokens = orig.split()               # para saber cómo venía cada palabra
    fixed = []

    for i, tok in enumerate(tokens):
        pre = "".join(ch for ch in tok if ch in "¿¡(\"'“”‘’[{-")
        suf = "".join(ch for ch in tok if ch in ").,:;!?\"'”’]}-%")
        core = tok.strip("¿¡(\"'“”‘’[{-).,:;!?”’]}-%")

        orig_tok = orig_tokens[i] if i < len(orig_tokens) else tok
        orig_core = orig_tok.strip("¿¡(\"'“”‘’[{-).,:;!?”’]}-%")
        core_up = core.upper()

        # Regla de siglas:
        # 1) Si está en lista ACRONYMS -> MAYÚS
        # 2) Si en el original era MAYÚS y parece sigla (2-5 letras) -> MAYÚS
        if (core_up in ACRONYMS) or (orig_core.isalpha() and 2 <= len(orig_core) <= 5 and orig_core.isupper()):
            new_core = core_up
        else:
            new_core = core  # ya viene en minúsculas por 'base'

        fixed.append(f"{pre}{new_core}{suf}")

    # Garantiza que la primera letra visible esté en mayúscula
    out = " ".join(fixed)
    for i, ch in enumerate(out):
        if ch.isalpha():
            out = out[:i] + out[i].upper() + out[i+1:]
            break

    return out

# ---------- Publicación ----------
def publish_one_for_category(conn, category_name, publish_status="publish"):
    cat_id = get_or_create_category_id_exact(category_name)
    for entry in pick_entry_for_category(category_name):
        title = normalize_title_case(entry.get("title") or "(Sin título)")
        link = entry.get("link")
        if not link or already_posted(conn, link):
            continue
        try:
            html, final_url = fetch_url(link)
        except Exception as e:
            print(f"[WARN] No se pudo abrir {link}: {e}")
            continue

        media_id = None
        try:
            cover = extract_og_image(html, final_url)
            if cover:
                media_id = wp_upload_media(to_jpeg_bytes(cover), "portada.jpg")
        except Exception as e:
            print(f"[WARN] Imagen falló: {e}")

        if POST_MODE == "full_html":
            soup = BeautifulSoup(Document(html).summary(), "html.parser")
            soup = strip_author_nodes(soup)
            for tag in soup.find_all(["img","picture","source","figure","figcaption"]):
                tag.decompose()
            content_html = str(soup)
        else:
            summary = summarize_html(html, max_words=1000)
            content_html = build_post_html(summary)

        try:
            post_id, post_link = wp_create_post(
                title, content_html,
                featured_media_id=media_id,
                status=publish_status,
                category_id=cat_id
            )
            mark_posted(conn, link)
            print(f"[OK] {category_name} ({publish_status}): {post_link or post_id}")
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
    ok, cat, idx, nxt = run_rotating_once(publish_status="publish")
    print(f"[ROTATION] idx={idx} cat={cat} -> next={nxt}")
