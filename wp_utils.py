import requests

class WordPressAPI:
    def __init__(self, url, user, app_password):
        self.url = url.rstrip("/")
        self.auth = (user, app_password)
        self.api_bases = [
            f"{self.url}/wp-json/wp/v2",
            f"{self.url}/?rest_route=/wp/v2"
        ]
        self.headers_json = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "NewsBot/1.0"
        }
        self.headers_bin = {
            "Accept": "application/json",
            "User-Agent": "NewsBot/1.0"
        }

    # ---------- Helpers ----------
    def _post_with_redirect(self, url, *, json=None, data=None, headers=None, max_hops=3):
        current = url
        for _ in range(max_hops + 1):
            r = requests.post(current, json=json, data=data, headers=headers, auth=self.auth,
                              timeout=30, allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location")
                if not loc:
                    r.raise_for_status()
                current = loc
                continue
            return r
        raise RuntimeError("Demasiadas redirecciones en POST")

    # ---------- Categorías ----------
    def get_categories_by_search(self, name):
        for base in self.api_bases:
            url = f"{base}/categories?per_page=100&search={name}"
            r = requests.get(url, headers=self.headers_json, auth=self.auth, timeout=20)
            if r.ok:
                return r.json()
        return []

    def create_category(self, name, parent=None):
        payload = {"name": name}
        if parent:
            payload["parent"] = parent
        for base in self.api_bases:
            r = self._post_with_redirect(f"{base}/categories", json=payload, headers=self.headers_json)
            if r.ok:
                return r.json()["id"]
        raise RuntimeError(f"No se pudo crear categoría {name}")

    def get_or_create_category_id_exact(self, name):
        cats = self.get_categories_by_search(name)
        for c in cats:
            if c.get("name", "").strip().lower() == name.strip().lower():
                return c["id"]
        return self.create_category(name)

    # ---------- Posts ----------
    def create_post(self, title, html_content, featured_media_id=None, status="draft",
                    category_id=None, excerpt=None, slug=None):
        payload = {
            "title": title,
            "content": html_content,
            "status": status
        }
        if excerpt:
            payload["excerpt"] = excerpt
        if slug:
            payload["slug"] = slug
        if category_id:
            payload["categories"] = [category_id]
        if featured_media_id:
            payload["featured_media"] = featured_media_id

        for base in self.api_bases:
            r = self._post_with_redirect(f"{base}/posts/", json=payload, headers=self.headers_json)
            if r.ok:
                data = r.json()
                return data["id"], data.get("link")
        raise RuntimeError("No se pudo crear post")

    # ---------- Media ----------
    def upload_media(self, jpeg_bytes, filename):
        headers = {
            **self.headers_bin,
            "Content-Type": "image/jpeg",
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        for base in self.api_bases:
            r = self._post_with_redirect(f"{base}/media/", data=jpeg_bytes, headers=headers)
            if r.ok:
                return r.json()["id"]
        raise RuntimeError("No se pudo subir media")
