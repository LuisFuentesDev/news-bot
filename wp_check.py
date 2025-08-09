import os
import requests
from dotenv import load_dotenv

load_dotenv()

WP_URL = os.getenv("WP_URL", "").rstrip("/")
WP_USER = os.getenv("WP_USER")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")

assert WP_URL and WP_USER and WP_APP_PASSWORD, "Faltan WP_URL/WP_USER/WP_APP_PASSWORD en .env"

resp = requests.get(f"{WP_URL}/wp-json/wp/v2/users/me", auth=(WP_USER, WP_APP_PASSWORD), timeout=20)
print("Status:", resp.status_code)
if resp.ok:
    data = resp.json()
    print("Conectado como:", data.get("name") or data.get("slug"))
else:
    print("Respuesta:", resp.text[:500])
