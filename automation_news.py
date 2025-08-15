import os
import re
import io
import unicodedata
from urllib.parse import urljoin

import requests
import feedparser
from bs4 import BeautifulSoup
from readability import Document
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from wp_utils import WordPressAPI
from db_utils import init_db, already_posted, mark_posted
from twitter_utils import can_tweet, record_tweet_success, build_tweet_text, tweet_news

load_dotenv()

# ---------- Configuración ----------
WP_URL = os.getenv("WP_URL", "").rstrip("/")
WP_USER = os.getenv("WP_USER")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")
POST_MODE = os.getenv("POST_MODE", "summary")  # summary | full_html
LANG = os.getenv("LANG", "es")

CATEGORY_ORDER = ["REGIONAL", "NACIONAL", "INTERNACIONAL", "DEPORTES"]

FEEDS_BY_CATEGORY = {
    "REGIONAL": ["https://www.soychile.cl/rss"],
    "NACIONAL": ["https://www.emol.com/rss/rss.asp"],
    "INTERNACIONAL": ["https://cnnespanol.cnn.com/feed/"],
    "DEPORTES": ["https://www.ole.com.ar/rss/ultimas-noticias/"],
}

HEADERS = {
    "User-Agent": "NewsBot/1.0 (+winforma.cl)",
    "Accept": "text/html,application/xhtml+xml"
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Helpers ----------
def make_seo_title(title: str, max_len=500) -> str:
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

def build_post_html(paragraphs):
    return "<div class='winf-body' style='padding-top:30px; font-size:18px; line-height:1.6;'>" + "".join(
        f"<p style='text-align:justify; margin-top: 15px;'>{p.strip()}</p>"
        for p in paragraphs if p.strip()
    ) + "</div>"

# ---------- GPT rewrite ----------
def rewrite_with_gpt(title, paragraphs):
    try:
        prompt = (
            f"Reescribe el titular y el cuerpo de la noticia sin inventar datos. "
            f"Devuelve JSON con 'title' y 'body'.\nTítulo: {title}\nCuerpo:\n" + "\n".join(paragraphs)
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un periodista profesional."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(resp.choices[0].message.content.strip())
        return data.get("title", title).strip(), data.get("body", build_post_html(paragraphs)).strip()
    except Exception as e:
        print(f"[WARN] GPT rewrite falló: {e}")
        return title, build_post_html(paragraphs or [])

# ---------- HTML parsing ----------
def fetch_url(url, timeout=20):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text, r.url

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
    scale = max(target_w / im.width, target_h / im.height)
    new = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    left = (new.width - target_w) // 2
    top  = (new.height - target_h) // 2
    new = new.crop((left, top, left + target_w, top + target_h))
    buf = io.BytesIO()
    new.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
    buf.seek(0)
    return buf.read()

def summarize_html(html, max_words=1000):
    doc = Document(html)
    soup = BeautifulSoup(doc.summary(), "html.parser")
    for bad in soup(["script", "style", "aside", "footer", "nav", "figure", "figcaption", "noscript", "img", "picture", "source"]):
        bad.decompose()
    total_words = 0
    clean_paragraphs = []
    for p in soup.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
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

# ---------- Publicación ----------
def publish_one_for_category(conn, wp, category_name, publish_status="publish"):
    cat_id = wp.get_or_create_category_id_exact(category_name)

    for entry in feedparser.parse(FEEDS_BY_CATEGORY[category_name][0]).entries:
        title = entry.get("title") or "(Sin título)"
        link = entry.get("link")
        if not link or already_posted(conn, link):
            continue

        html, final_url = fetch_url(link)
        media_id = None
        cover = extract_og_image(html, final_url)
        if cover:
            media_id = wp.upload_media(to_jpeg_bytes(cover), "portada.jpg")

        paragraphs = summarize_html(html, max_words=1000)
        seo_title = make_seo_title(title)
        meta_desc = make_meta_description(paragraphs)
        slug = slugify(title)

        new_title, new_body = rewrite_with_gpt(seo_title, paragraphs)

        post_id, post_link = wp.create_post(
            new_title, new_body, featured_media_id=media_id,
            status=publish_status, category_id=cat_id,
            excerpt=meta_desc, slug=slug
        )

        mark_posted(conn, final_url)
        print(f"[OK] {category_name}: {post_link or post_id}")

        if publish_status == "publish" and post_link and can_tweet(conn):
            tweet_text = build_tweet_text(new_title, post_link, category_name)
            if tweet_news(tweet_text):
                used = record_tweet_success(conn)
                print(f"[X] Tweets usados: {used}")
        return True
    return False

# ---------- Rotación ----------
def run_rotating_once(publish_status="publish"):
    conn = init_db()
    wp = WordPressAPI(WP_URL, WP_USER, WP_APP_PASSWORD)
    try:
        idx = int(conn.execute("SELECT v FROM state WHERE k='rotation_idx'").fetchone() or (0,))[0] % len(CATEGORY_ORDER)
        cat = CATEGORY_ORDER[idx]
        ok = publish_one_for_category(conn, wp, cat, publish_status=publish_status)
        conn.execute(
            "INSERT OR REPLACE INTO state(k,v) VALUES('rotation_idx',?)",
            ((idx + 1) % len(CATEGORY_ORDER),)
        )
        conn.commit()
        return ok, cat, idx, (idx + 1) % len(CATEGORY_ORDER)
    finally:
        conn.close()

# ---------- Main ----------
if __name__ == "__main__":
    ok, cat, idx, nxt = run_rotating_once(publish_status="publish")
    print(f"[ROTATION] idx={idx} cat={cat} -> next={nxt}")
