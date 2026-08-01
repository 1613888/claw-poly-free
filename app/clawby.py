"""Polymarket public data client (free, no API key).

Replaces the former Clawby paid relay (which returned HTTP 402 Payment
Required). Two data paths, both public and authentication-free:
  - events metadata  -> Gamma API  https://gamma-api.polymarket.com/events
  - orderbook quotes -> CLOB API   https://clob.polymarket.com/book
BTC spot price is handled separately in btc.py via Binance (also free).

A browser User-Agent is mandatory: Polymarket's edge blocks the default
python/httpx User-Agent with HTTP 403. The global throttle + 5xx retry are
kept so existing callers (engine, executor, markets, redeem) work unchanged.
"""
import asyncio
import json
import logging
import time

import httpx

from . import config

log = logging.getLogger("clawby")

_MIN_GAP = 0.1
CALLS = 0
CALLS_BY = {}
_last = 0.0
_gap = asyncio.Lock()

# Polymarket's edge 403s the default httpx UA; a desktop browser UA is required.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
GAMMA_HOST = "https://gamma-api.polymarket.com"


async def validate_key(key):
    """No-op kept for admin compatibility. The public data sources need no key."""
    return True, "已切换为 Polymarket/Binance 免费公共数据源,无需 API Key"


async def _get_json(url, params=None, timeout=30, retries=3):
    """GET JSON from a Polymarket public endpoint with a browser UA + 5xx retry."""
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    url, params=params,
                    headers={"User-Agent": _UA, "Accept": "application/json"})
                if resp.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                    log.warning("GET %s -> %s, backoff %ds", url,
                                resp.status_code, 2 * (attempt + 1))
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt == retries - 1:
                raise
            log.warning("GET %s retry %d: %s", url, attempt + 1, exc)
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def relay(name, params=None, timeout=30, retries=3):
    """Dispatch a former Clawby relay name to a Polymarket public endpoint.

    Supported names (no API key, no auth):
      polymarket_events    -> Gamma /events?slug=...        (returns a list)
      polymarket_orderbook -> CLOB  /book?token_id=...      (returns a dict)
    Returns the raw JSON; the Clawby envelope unwrapping is gone since the
    public endpoints respond directly. CALLS counters are kept for the
    engine's slow-tick instrumentation.
    """
    global _last, CALLS
    CALLS += 1
    CALLS_BY[name] = CALLS_BY.get(name, 0) + 1
    params = params or {}
    async with _gap:                       # global throttle (>= _MIN_GAP apart)
        wait = _MIN_GAP - (time.monotonic() - _last)
        if wait > 0:
            await asyncio.sleep(wait)
        _last = time.monotonic()
    if name == "polymarket_events":
        slug = params.get("slug")
        if not slug:
            raise RuntimeError("polymarket_events requires 'slug'")
        return await _get_json(f"{GAMMA_HOST}/events",
                               params={"slug": slug}, timeout=timeout, retries=retries)
    if name == "polymarket_orderbook":
        token_id = params.get("token_id")
        if not token_id:
            raise RuntimeError("polymarket_orderbook requires 'token_id'")
        return await _get_json(f"{config.CLOB_HOST}/book",
                               params={"token_id": token_id},
                               timeout=timeout, retries=retries)
    raise RuntimeError(f"unsupported relay name: {name}")


async def relay_safe(name, params=None, timeout=30):
    try:
        return await relay(name, params, timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("relay %s failed: %s", name, exc)
        return None


async def market_by_slug(slug):
    """-> {token_up, token_down, tick, accepting, neg_risk} or None."""
    ev = await relay_safe("polymarket_events", {"slug": slug})
    rows = ev if isinstance(ev, list) else (ev or {}).get("data") or []
    if not rows:
        return None
    m = (rows[0].get("markets") or [{}])[0]
    tokens = m.get("clobTokenIds")
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except ValueError:
            tokens = None
    if not tokens or len(tokens) < 2:
        return None
    try:
        tick = float(m.get("orderPriceMinTickSize") or 0.01)
    except (TypeError, ValueError):
        tick = 0.01
    return {"token_up": str(tokens[0]), "token_down": str(tokens[1]),
            "tick": tick, "accepting": bool(m.get("acceptingOrders")),
            "neg_risk": bool(m.get("negRisk")),
            "condition_id": m.get("conditionId") or ""}


_PX_CACHE = {}                     # token -> (monotonic_ts, result); shared by
_PX_TTL = 1                        # engine + admin pages to halve relay load


async def best_prices(token_id):
    """-> {bid, ask, mid} implied probabilities (None on failure). 8s TTL cache."""
    hit = _PX_CACHE.get(token_id)
    if hit and time.monotonic() - hit[0] < _PX_TTL:
        return hit[1]
    ob = await relay_safe("polymarket_orderbook", {"token_id": token_id})
    if not isinstance(ob, dict):
        return None
    try:
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        bid = max(float(b["price"]) for b in bids) if bids else None
        ask = min(float(a["price"]) for a in asks) if asks else None
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        out = {"bid": bid, "ask": ask, "mid": mid}
        _PX_CACHE[token_id] = (time.monotonic(), out)
        if len(_PX_CACHE) > 600:
            _PX_CACHE.clear()
        return out
    except (TypeError, ValueError, KeyError):
        return None
