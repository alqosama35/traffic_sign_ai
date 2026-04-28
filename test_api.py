"""
FR-API-06: test_api.py
Sends 10+ representative test images to /predict and asserts:
  - HTTP status is 200
  - 'class' field is a non-empty string
  - 'confidence' is between 0 and 1
  - 'top_3' contains exactly 3 entries
  - each top_3 entry has 'class' (str) and 'confidence' (float in [0,1])
  - 'note' disclaimer is present
  - GET /health returns 200 {"status": "ok"}

Images are generated in memory as valid JPEG files — no external files needed.
This makes the test fully self-contained and safe to run in CI.
"""

import io
import sys
import time

import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"

# ── Test image factory ────────────────────────────────────────────────────────
# Generates synthetic but valid 224×224 JPEG images in memory.
# Each image has a different color so they are meaningfully distinct.
# The model will produce real softmax outputs on any valid RGB image.
COLORS = [
    (220, 50,  50),   # red
    (50,  220, 50),   # green
    (50,  50,  220),  # blue
    (220, 220, 50),   # yellow
    (220, 50,  220),  # magenta
    (50,  220, 220),  # cyan
    (180, 100, 50),   # orange-brown
    (100, 50,  180),  # purple
    (200, 200, 200),  # light grey
    (60,  60,  60),   # dark grey
    (255, 255, 255),  # white
    (10,  10,  10),   # near-black
]


def make_image_bytes(color: tuple[int, int, int]) -> bytes:
    """Return JPEG bytes for a 224×224 solid-color image with a centred circle."""
    img = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    # Draw a white/dark circle to add some feature variation
    contrast = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    draw.ellipse([60, 60, 164, 164], outline=contrast, width=8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ── Helpers ───────────────────────────────────────────────────────────────────
def assert_eq(label: str, actual, expected):
    if actual != expected:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        return False
    return True


def assert_true(label: str, condition: bool, detail: str = ""):
    if not condition:
        print(f"  FAIL  {label}{': ' + detail if detail else ''}")
        return False
    return True


# ── Test suite ────────────────────────────────────────────────────────────────
def run_tests():
    failures = 0
    total = 0

    # ── Test 0: /health ───────────────────────────────────────────────────────
    print("\n[0] GET /health")
    total += 1
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        ok = assert_eq("status code", r.status_code, 200)
        ok = assert_eq("body", r.json(), {"status": "ok"}) and ok
        if ok:
            print("  PASS")
        else:
            failures += 1
    except Exception as exc:
        print(f"  FAIL  exception: {exc}")
        failures += 1

    # ── Tests 1-12: POST /predict ─────────────────────────────────────────────
    for i, color in enumerate(COLORS, start=1):
        print(f"\n[{i}] POST /predict  color=rgb{color}")
        total += 1
        ok = True

        image_bytes = make_image_bytes(color)
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}

        start = time.monotonic()
        try:
            r = requests.post(f"{BASE_URL}/predict", files=files, timeout=15)
        except Exception as exc:
            print(f"  FAIL  exception: {exc}")
            failures += 1
            continue
        elapsed_ms = (time.monotonic() - start) * 1000

        # FR-API-06 assertions
        ok = assert_eq("status code", r.status_code, 200) and ok

        if r.status_code != 200:
            print(f"  response body: {r.text}")
            failures += 1
            continue

        body = r.json()

        # 'class' must be a non-empty string
        ok = assert_true(
            "'class' is str",
            isinstance(body.get("class"), str) and len(body.get("class", "")) > 0,
            f"got: {body.get('class')!r}",
        ) and ok

        # 'confidence' must be float in [0, 1]
        conf = body.get("confidence")
        ok = assert_true(
            "confidence in [0,1]",
            isinstance(conf, float) and 0.0 <= conf <= 1.0,
            f"got: {conf!r}",
        ) and ok

        # 'top_3' must have exactly 3 entries
        top3 = body.get("top_3", [])
        ok = assert_eq("len(top_3)", len(top3), 3) and ok

        # Each top_3 entry must have 'class' (str) and 'confidence' (float in [0,1])
        for j, entry in enumerate(top3):
            ok = assert_true(
                f"top_3[{j}].class is str",
                isinstance(entry.get("class"), str),
                f"got: {entry.get('class')!r}",
            ) and ok
            c = entry.get("confidence")
            ok = assert_true(
                f"top_3[{j}].confidence in [0,1]",
                isinstance(c, (int, float)) and 0.0 <= c <= 1.0,
                f"got: {c!r}",
            ) and ok

        # 'note' disclaimer must be present — FR-API-03
        ok = assert_true(
            "'note' disclaimer present",
            isinstance(body.get("note"), str) and len(body.get("note", "")) > 0,
            f"got: {body.get('note')!r}",
        ) and ok

        # FR-API-07: response time < 500ms (informational — does not fail the test)
        latency_ok = elapsed_ms < 500
        print(f"  latency: {elapsed_ms:.0f}ms {'✓' if latency_ok else '⚠ > 500ms'}")

        if ok:
            print("  PASS")
        else:
            failures += 1

    # ── Test 13: invalid file should return 422 ───────────────────────────────
    print("\n[13] POST /predict  invalid file (not an image)")
    total += 1
    files = {"file": ("bad.txt", b"this is not an image", "text/plain")}
    try:
        r = requests.post(f"{BASE_URL}/predict", files=files, timeout=10)
        ok = assert_true(
            "status code is 422",
            r.status_code == 422,
            f"got: {r.status_code}",
        )
        if ok:
            print("  PASS")
        else:
            failures += 1
    except Exception as exc:
        print(f"  FAIL  exception: {exc}")
        failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print(f"Results: {total - failures}/{total} passed")
    if failures:
        print(f"FAILED: {failures} test(s) failed.")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()