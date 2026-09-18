"""One place that builds HTTP clients, so TLS configuration cannot drift.

**Why this module exists.** httpx verifies TLS against the `certifi` bundle,
which contains only public roots. On a corporate network doing TLS inspection —
this machine sits behind Zscaler — every certificate is re-signed by a private
CA that `certifi` has never heard of, so *every* httpx call fails with
``CERTIFICATE_VERIFY_FAILED``. The fix is to verify against
``ssl.create_default_context()`` instead, which loads the operating system's
own trust store, where the proxy's CA actually lives.

This is not a weakening of verification: certificates are still fully verified,
just against the OS roots rather than a bundled copy. It is also strictly more
portable — on Linux and macOS the same call resolves to the normal OpenSSL
paths.
"""

from __future__ import annotations

import ssl
from functools import lru_cache

import httpx

# Cloudflare fronts DexScreener and answers the stock httpx user-agent with a
# 403 interstitial whenever it feels like it. A plain browser UA is enough;
# nothing else about the request needs to lie.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json"}


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    """The OS trust store. Cached — building a context reads the whole root
    store from disk, and doing that per request is measurably slow."""
    return ssl.create_default_context()


def make_client(
    timeout: float = 15.0, headers: dict[str, str] | None = None
) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        headers={**DEFAULT_HEADERS, **(headers or {})},
        verify=ssl_context(),
        follow_redirects=True,
    )
