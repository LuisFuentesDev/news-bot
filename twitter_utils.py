import os
import datetime as dt
import tweepy
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from db_utils import get_state, set_state

# ---------- Configuración ----------
TW_ENABLED = os.getenv("TW_ENABLED", "1")  # "1" para habilitar
TW_MONTHLY_LIMIT = int(os.getenv("TW_MONTHLY_LIMIT", "500"))  # Límite de tweets por mes

# Hashtags por categoría
HASHTAGS = {
    "REGIONAL": ["#Chile", "#Regional"],
    "NACIONAL": ["#Chile", "#Nacional"],
    "INTERNACIONAL": ["#Internacional"],
    "DEPORTES": ["#Deportes"],
}

# ---------- URL con UTM ----------
def add_utm(u: str, **utm) -> str:
    """Agrega parámetros UTM a una URL."""
    s = urlsplit(u)
    q = dict(parse_qsl(s.query, keep_blank_values=True))
    q.update(utm)
    return urlunsplit((s.scheme, s.netloc, s.path, urlencode(q), s.fragment))

# ---------- Contador mensual ----------
def _month_key_utc():
    """Clave única para el mes actual en UTC."""
    return dt.datetime.utcnow().strftime("%Y-%m")

def _ensure_month(conn):
    """Reinicia contador si cambiamos de mes."""
    mk = _month_key_utc()
    curr = get_state(conn, "tw_month", "")
    if curr != mk:
        set_state(conn, "tw_month", mk)
        set_state(conn, "tw_writes", "0")

def get_tw_writes(conn) -> int:
    """Obtiene el número de tweets enviados este mes."""
    _ensure_month(conn)
    v = get_state(conn, "tw_writes", "0") or "0"
    try:
        return int(v)
    except ValueError:
        set_state(conn, "tw_writes", "0")
        return 0

def can_tweet(conn) -> bool:
    """Verifica si podemos twittear según el límite mensual."""
    _ensure_month(conn)
    count = get_tw_writes(conn)
    return TW_ENABLED == "1" and count < TW_MONTHLY_LIMIT

def record_tweet_success(conn) -> int:
    """Incrementa contador tras un tweet exitoso."""
    c = get_tw_writes(conn) + 1
    set_state(conn, "tw_writes", str(c))
    return c

# ---------- Construir tweet ----------
def build_tweet_text(title: str, url: str, cat: str) -> str:
    """Genera texto de tweet con hashtags y UTM."""
    url_utm = add_utm(url, utm_source="x", utm_medium="social", utm_campaign="newsbot")
    tags = HASHTAGS.get(cat, [])
    base = f"{title}\n{url_utm}"
    if tags:
        tail = " " + " ".join(tags[:2])
        if len(base) + len(tail) <= 280:
            base += tail
    return base[:280]

# ---------- Enviar tweet ----------
def tweet_news(text: str) -> bool:
    """Publica un tweet en X (Twitter) usando Tweepy."""
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
