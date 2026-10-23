"""
Cliente Supabase para el motor de señales.
Usa la service_role key para poder escribir en la tabla signals.
"""
import os
from supabase import create_client, Client

_client: Client | None = None


def get_supabase() -> Client | None:
    """
    Devuelve el cliente Supabase (singleton).
    Si las variables de entorno no están configuradas, devuelve None
    y el motor sigue funcionando sin Supabase (degradación elegante).
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        return None

    try:
        _client = create_client(url, key)
        return _client
    except Exception:
        return None
