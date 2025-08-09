import os
import requests
from dotenv import load_dotenv

load_dotenv()

WP_URL = os.getenv("WP_URL", "").rstrip("/")
WP_USER = os.getenv("WP_USER")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")
WP_CATEGORY_ID = int(os.getenv("WP_CATEGORY_ID", "0") or 0)

assert WP_URL and WP_USER and WP_APP_PASSWORD, "Faltan WP_URL/WP_USER/WP_APP_PASSWORD en .env"

auth = (WP_USER, WP_APP_PASSWORD)

# Opción A (normal). Si diera problemas, cambia a opción B:
API_BASE = f"{WP_URL}/wp-json/wp/v2"
# Opción B (fallback):
# API_BASE = f"{WP_URL}/?rest_route=/wp/v2"

POSTS_URL = f"{API_BASE}/posts/"
MEDIA_URL = f"{API_BASE}/media/"

HEADERS_JSON = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "NewsBot/1.0"
}
HEADERS_BIN = {
    "Accept": "application/json",
    "User-Agent": "NewsBot/1.0"
}

def _post_with_redirect(url, *, json=None, data=None, headers=None, max_hops=3):
    """Reintenta manualmente POST si hay 301/302/303/307/308 con header Location."""
    current_url = url
    for hop in range(max_hops + 1):
        r = requests.post(current_url, json=json, data=data,
                          headers=headers, auth=auth, timeout=30,
                          allow_redirects=False)
        print(f"POST {current_url} -> {r.status_code} {r.headers.get('Location','')}")
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("Location")
            if not loc:
                r.raise_for_status()
            current_url = loc
            continue
        return r
    raise RuntimeError("Demasiadas redirecciones en POST")

def create_post(title, html_content, featured_media_id=None, status="draft"):
    payload = {
        "title": title,
        "content": html_content,
        "status": status,
    }
    if WP_CATEGORY_ID:
        payload["categories"] = [WP_CATEGORY_ID]
    if featured_media_id:
        payload["featured_media"] = featured_media_id

    r = _post_with_redirect(POSTS_URL, json=payload, headers=HEADERS_JSON)
    print("Body sample:", r.text[:280].replace("\n", " "), "\n")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        raise RuntimeError("El servidor devolvió una LISTA (parece que el POST fue transformado en GET).")
    return data["id"], data.get("link")

def upload_media_from_url(image_url, filename="portada.jpg"):
    img = requests.get(image_url, headers=HEADERS_BIN, timeout=20)
    img.raise_for_status()
    headers = {
        **HEADERS_BIN,
        "Content-Type": "image/jpeg",
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    r = _post_with_redirect(MEDIA_URL, data=img.content, headers=headers)
    print("Body sample:", r.text[:200].replace("\n", " "), "\n")
    r.raise_for_status()
    return r.json()["id"]

if __name__ == "__main__":
    # A) Crear un borrador
    post_id, link = create_post(
        title="(Prueba) Publicación desde script",
        html_content="<p>Hola, esto es una prueba creada vía REST API desde Python.</p>",
        status="draft"
    )
    print("Borrador creado:", post_id, link)

    # B) Subir imagen y asignarla como destacada
    prueba_img = "https://picsum.photos/1200/630.jpg"
    media_id = upload_media_from_url(prueba_img, filename="portada_prueba.jpg")
    print("Media subida con ID:", media_id)

    # C) Fijar imagen destacada
    update_url = f"{API_BASE}/posts/{post_id}"
    r = _post_with_redirect(update_url, json={"featured_media": media_id}, headers=HEADERS_JSON)
    print("Update featured_media ->", r.status_code)
    print("Body sample:", r.text[:200].replace("\n", " "), "\n")
    r.raise_for_status()
    print("Imagen destacada asignada al post:", post_id)
