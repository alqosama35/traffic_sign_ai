"""
FR-API-06: test_api.py
All assertions per SRS:
  - /health returns 200 {"status":"ok"}                   (FR-API-02)
  - /predict returns 200                                  (FR-API-01)
  - 'class' is a non-empty string                         (FR-API-01)
  - 'category' is a non-empty string                      (FR-API-01)
  - 'confidence' is float in [0,1]                        (FR-API-01)
  - 'top_3' has exactly 3 entries                         (FR-API-01)
  - each top_3 entry has 'class' (str) + 'confidence'     (FR-API-01)
  - 'note' disclaimer is present                          (FR-API-03)
  - invalid image returns 422                             (FR-API-01)
  - wrong API key returns 401 or 200 in dev mode          (NFR-SEC-01)
  - latency < 500ms reported (informational, not a failure)(FR-API-07)
"""

import io
import os
import sys
import time

import requests
from PIL import Image, ImageDraw

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.environ.get("API_KEY", "")   # empty = DEV_MODE, skip auth header

COLORS = [
    (220, 50,  50),  (50,  220, 50),  (50,  50,  220),
    (220, 220, 50),  (220, 50,  220), (50,  220, 220),
    (180, 100, 50),  (100, 50,  180), (200, 200, 200),
    (60,  60,  60),  (255, 255, 255), (10,  10,  10),
]


def make_image_bytes(color: tuple) -> bytes:
    img  = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    contrast = (0,0,0) if sum(color) > 380 else (255,255,255)
    draw.ellipse([60,60,164,164], outline=contrast, width=8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _headers() -> dict:
    """Only include X-API-Key when API_KEY is set (prod). Omit in dev mode."""
    return {"X-API-Key": API_KEY} if API_KEY else {}


def assert_eq(label, actual, expected):
    if actual != expected:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        return False
    return True


def assert_true(label, cond, detail=""):
    if not cond:
        print(f"  FAIL  {label}{': '+detail if detail else ''}")
        return False
    return True


def run_tests():
    failures = 0
    total    = 0

    # ── Test 0: GET /health ───────────────────────────────────────────────────
    print("\n[0] GET /health")
    total += 1
    try:
        r  = requests.get(f"{BASE_URL}/health", timeout=10)
        ok = assert_eq("status code", r.status_code, 200)
        ok = assert_eq("body",        r.json(), {"status": "ok"}) and ok
        print("  PASS" if ok else "  FAIL")
        if not ok:
            failures += 1
    except Exception as exc:
        print(f"  FAIL  exception: {exc}")
        failures += 1

    # ── Tests 1-12: POST /predict ─────────────────────────────────────────────
    for i, color in enumerate(COLORS, start=1):
        print(f"\n[{i}] POST /predict  color=rgb{color}")
        total += 1
        ok    = True

        files = {"file": ("test.jpg", make_image_bytes(color), "image/jpeg")}
        t0    = time.monotonic()
        try:
            r = requests.post(f"{BASE_URL}/predict", files=files,
                              headers=_headers(), timeout=15)
        except Exception as exc:
            print(f"  FAIL  exception: {exc}")
            failures += 1
            continue

        elapsed_ms = (time.monotonic() - t0) * 1000

        ok = assert_eq("status code", r.status_code, 200) and ok
        if r.status_code != 200:
            print(f"  response: {r.text}")
            failures += 1
            continue

        body = r.json()

        # FR-API-01: class
        ok = assert_true("'class' is non-empty str",
                         isinstance(body.get("class"), str) and len(body.get("class","")) > 0,
                         f"got: {body.get('class')!r}") and ok

        # FR-API-01: category (Issue 6 fix)
        ok = assert_true("'category' is non-empty str",
                         isinstance(body.get("category"), str) and len(body.get("category","")) > 0,
                         f"got: {body.get('category')!r}") and ok

        # FR-API-01: confidence
        conf = body.get("confidence")
        ok = assert_true("confidence in [0,1]",
                         isinstance(conf, float) and 0.0 <= conf <= 1.0,
                         f"got: {conf!r}") and ok

        # FR-API-01: top_3 length
        top3 = body.get("top_3", [])
        ok = assert_eq("len(top_3)", len(top3), 3) and ok

        # FR-API-01: top_3 entries
        for j, entry in enumerate(top3):
            ok = assert_true(f"top_3[{j}].class is str",
                             isinstance(entry.get("class"), str)) and ok
            c  = entry.get("confidence")
            ok = assert_true(f"top_3[{j}].confidence in [0,1]",
                             isinstance(c, (int,float)) and 0.0 <= c <= 1.0,
                             f"got: {c!r}") and ok

        # FR-API-03: note disclaimer
        ok = assert_true("'note' disclaimer present",
                         isinstance(body.get("note"), str) and len(body.get("note","")) > 0,
                         f"got: {body.get('note')!r}") and ok

        # FR-API-07: latency (informational — does NOT fail the test)
        latency_ok = elapsed_ms < 500
        print(f"  latency: {elapsed_ms:.0f}ms {'✓' if latency_ok else '⚠ above 500ms (CPU only — acceptable locally)'}")

        print("  PASS" if ok else "  FAIL")
        if not ok:
            failures += 1

    # ── Test 13: invalid image → 422 ─────────────────────────────────────────
    print("\n[13] POST /predict  invalid file → expect 422")
    total += 1
    files = {"file": ("bad.txt", b"not an image", "text/plain")}
    try:
        r  = requests.post(f"{BASE_URL}/predict", files=files,
                           headers=_headers(), timeout=10)
        ok = assert_true("status 422", r.status_code == 422, f"got: {r.status_code}")
        print("  PASS" if ok else "  FAIL")
        if not ok:
            failures += 1
    except Exception as exc:
        print(f"  FAIL  exception: {exc}")
        failures += 1

    # ── Test 14: wrong API key → 401 (Issue 7 fix) ───────────────────────────
    # In DEV_MODE (API_KEY not set), auth is skipped so we expect 200.
    # In production (API_KEY set), a wrong key must return 401.
    print("\n[14] POST /predict  wrong API key → expect 401 (or 200 in dev mode)")
    total += 1
    files = {"file": ("test.jpg", make_image_bytes((128,128,128)), "image/jpeg")}
    try:
        r  = requests.post(f"{BASE_URL}/predict", files=files,
                           headers={"X-API-Key": "definitely-wrong-key"}, timeout=10)
        if API_KEY:
            # Production: must reject
            ok = assert_true("status 401", r.status_code == 401, f"got: {r.status_code}")
        else:
            # Dev mode: auth disabled, 200 is correct
            ok = assert_true("status 200 (dev mode)", r.status_code == 200, f"got: {r.status_code}")
        print("  PASS" if ok else "  FAIL")
        if not ok:
            failures += 1
    except Exception as exc:
        print(f"  FAIL  exception: {exc}")
        failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*45}")
    print(f"Results: {total - failures}/{total} passed")
    if failures:
        print(f"FAILED: {failures} test(s) — pipeline will block merge (FR-CI-04)")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()