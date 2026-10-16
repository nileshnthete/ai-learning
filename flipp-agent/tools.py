import re
from datetime import datetime
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
import requests

DEFAULT_POSTAL_CODE = "N2L3G1"     # Waterloo, ON - change to your own
FLIPP_URL = "https://backflipp.wishabi.com/flipp/items/search"


def to_local_date(iso_string):
    """Convert Flipp's UTC timestamp to a Toronto-time date like 'Oct 7'."""
    if not iso_string:
        return None
    try:
        utc_time = datetime.fromisoformat(iso_string)
        local_time = utc_time.astimezone(ZoneInfo("America/Toronto"))
        return local_time.strftime("%b %d").replace(" 0", " ")
    except ValueError:
        return iso_string  # unknown format: pass it through unchanged
        
# ---------- 1. The tool: plain Python that does the real work ----------

def _normalize(text):
    """'LACTANTIA PUR FILTRE MILK' -> 'lactantiapurfiltremilk'"""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())

def _price_number(text):
    """'5.98 EACH' -> 5.98"""
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return float(match.group()) if match else None

def is_duplicate(a, b):
    """Same store, same price, and names at least 85% similar."""
    if a["store"] != b["store"] or _price_number(a["price"]) != _price_number(b["price"]):
        return False
    return SequenceMatcher(None, _normalize(a["name"]), _normalize(b["name"])).ratio() > 0.85


def search_flipp(query: str, postal_code: str = DEFAULT_POSTAL_CODE, limit: int = 5):
    resp = requests.get(
        FLIPP_URL,
        params={"locale": "en-ca", "postal_code": postal_code.replace(" ", ""), "q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])          # no [:limit] here any more

    results = []
    for i in items:
        price_text = " ".join(
            str(p) for p in [i.get("pre_price_text"), i.get("current_price"), i.get("post_price_text")] if p
        )
        results.append({
            "name": i.get("name"),
            "store": i.get("merchant_name"),
            "price": price_text or "see flyer",
            "deal_ends": to_local_date(i.get("valid_to")),
        })

    unique = []
    for r in results:
        if not any(is_duplicate(r, u) for u in unique):
            unique.append(r)
    return unique[:limit]                          # limit applied after removing duplicates


UNIT_CONVERSIONS = {
    # unit as written -> (base unit, multiplier to convert into base unit)
    "l":  ("L", 1),
    "ml": ("L", 0.001),
    "kg": ("kg", 1),
    "g":  ("kg", 0.001),
    "lb": ("kg", 0.453592),
    # counts -> price per item
    "count":  ("each", 1),
    "ct":     ("each", 1),
    "pcs":    ("each", 1),
    "pc":     ("each", 1),
    "pieces": ("each", 1),
    "dozen":  ("each", 12),
}

def extract_size(text: str):
    """Find the first number+unit we recognize, e.g. '4 L', '500mL', '2 lb'."""
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", text or ""):
        number, unit = float(match.group(1)), match.group(2).lower()
        if unit in UNIT_CONVERSIONS and number > 0:
            return number, unit
    return None, None

def price_per_unit(price: float, size: str):
    number, unit = extract_size(size)
    if number is None:
        return {"error": f"Could not find a size like '4 L' or '500 g' in: {size!r}"}

    try:
        price = float(str(price).replace("$", "").strip())
    except ValueError:
        return {"error": f"Price must be a single number in dollars, got: {price!r}"}

    base_unit, multiplier = UNIT_CONVERSIONS[unit]
    return {
        "price_per_unit": round(price / (number * multiplier), 2),
        "unit": base_unit,
    }