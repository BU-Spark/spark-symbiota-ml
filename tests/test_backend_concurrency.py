"""POST /azure must isolate concurrent requests.

The handler previously wrote every download to a fixed temp.jpg in the working
directory, so overlapping requests overwrote each other and a caller could be
returned another caller's transcription. Under this test the old handler passed
1 of 8; a fixed filename fails it again immediately.

    python tests/test_backend_concurrency.py     # or: pytest tests/
"""
import hashlib
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend.main as m  # noqa: E402

# Slow enough that overlapping requests genuinely interleave.
DELAY = 0.05


def _png(seed):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (seed % 256, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, content):
        self.content = content
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_concurrent(endpoint):
    bodies = {f"http://example.invalid/{i}": _png(i * 37) for i in range(8)}
    expect = {u: hashlib.sha1(b).hexdigest() for u, b in bodies.items()}

    def fake_get(url, timeout=None, stream=False):
        threading.Event().wait(DELAY)
        return _Resp(bodies[url])

    # Echo a hash of the bytes that reached the pipeline; a crossed-over file
    # then shows up as a mismatched response.
    def fake_pipeline(path):
        threading.Event().wait(DELAY)
        return json.dumps({"scientificName": hashlib.sha1(open(path, "rb").read()).hexdigest(),
                           "confidence": {}})

    with patch.object(m, "requests") as rq, \
            patch.dict(m.PIPELINES, {k: fake_pipeline for k in m.PIPELINES}), \
            patch.object(m, "to_middleware_envelope", lambda d, model: d):
        rq.get = fake_get
        client = TestClient(m.app)
        with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            results = list(pool.map(
                lambda u: (u, client.post(endpoint, params={"url": u})), bodies))

    for url, resp in results:
        assert resp.status_code == 200, resp.text
        assert resp.json()["scientificName"] == expect[url], \
            f"{endpoint}: {url} got another request's image"


def test_concurrent_requests_do_not_share_a_temp_file():
    for endpoint in ("/azure", "/google", "/anthropic"):
        _run_concurrent(endpoint)


def test_every_pipeline_has_an_endpoint():
    client = TestClient(m.app)
    listed = {p["id"] for p in client.get("/pipelines").json()["pipelines"]}
    assert listed == set(m.PIPELINES), "pipelines endpoint disagrees with the registry"
    for name in m.PIPELINES:
        # 400 = handler reached and asked for input; 404 would mean unwired.
        assert client.post(f"/{name}").status_code == 400, f"/{name} not wired"


def test_upload_works_on_every_pipeline():
    body = _png(7)
    want = hashlib.sha1(body).hexdigest()

    def fake_pipeline(path):
        return json.dumps({"scientificName": hashlib.sha1(open(path, "rb").read()).hexdigest(),
                           "confidence": {}})

    with patch.dict(m.PIPELINES, {k: fake_pipeline for k in m.PIPELINES}), \
            patch.object(m, "to_middleware_envelope", lambda d, model: d):
        client = TestClient(m.app)
        for name in m.PIPELINES:
            r = client.post(f"/{name}", files={"file": ("sheet.jpg", body, "image/jpeg")})
            assert r.status_code == 200, f"/{name}: {r.text}"
            assert r.json()["scientificName"] == want


def test_upload_rejects_bad_input():
    client = TestClient(m.app)
    assert client.post("/azure").status_code == 400                     # neither url nor file
    assert client.post("/azure", files={"file": ("x.jpg", b"nope", "image/jpeg")}
                       ).status_code == 400                             # not an image
    assert client.post("/azure", files={"file": ("x.jpg", b"", "image/jpeg")}
                       ).status_code == 400                             # empty


def test_compare_runs_several_pipelines_on_one_upload():
    def fake_pipeline(path):
        return json.dumps({"scientificName": "Rumex acetosella", "barcode": "372522",
                           "confidence": {"scientificName": 0.95, "barcode": 0.97}})

    with patch.dict(m.PIPELINES, {k: fake_pipeline for k in m.PIPELINES}):
        client = TestClient(m.app)
        r = client.post("/compare?pipelines=azure,google",
                        files={"file": ("s.jpg", _png(3), "image/jpeg")})
        assert r.status_code == 200, r.text
        d = r.json()
        assert set(d["results"]) == {"azure", "google"}
        assert d["results"]["azure"]["ok"]
        # DWC keys, not the pipelines' internal names
        assert "catalogNumber" in d["results"]["azure"]["envelope"]
        assert set(d["ceilings"]) == {"azure", "google"}
        assert client.post("/compare?pipelines=nope",
                           files={"file": ("s.jpg", _png(3), "image/jpeg")}).status_code == 400


def test_original_routes_unchanged():
    # The tool is additive. GET / keeps its original JSON contract because the
    # middleware's deployment may health-check it.
    client = TestClient(m.app)
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["message"] == "OCR service is running"
    assert root.json()["pipelines"] == sorted(m.PIPELINES)
    assert client.get("/health").json()["pipelines"] == sorted(m.PIPELINES)
    page = client.get("/tool")
    assert page.status_code == 200 and "<title>" in page.text


def test_ceilings_are_probabilities():
    for p in TestClient(m.app).get("/pipelines").json()["pipelines"]:
        for field, ceiling in p["confidence_ceilings"].items():
            assert 0.0 < ceiling <= 1.0, f"{p['id']}.{field} ceiling {ceiling}"


def test_bad_image_is_rejected_without_leaking():
    def fake_get(url, timeout=None, stream=False):
        return _Resp(b"this is not an image")

    with patch.object(m, "requests") as rq:
        rq.get = fake_get
        resp = TestClient(m.app).post("/azure", params={"url": "http://example.invalid/x"})
    assert resp.status_code == 400


def test_review_logging(tmp_path=None):
    import tempfile
    original = m.REVIEWS
    m.REVIEWS = os.path.join(tempfile.mkdtemp(), "reviews.jsonl")
    try:
        client = TestClient(m.app)
        assert client.get("/reviews").json()["reviews_logged"] == 0
        r = client.post("/reviews", json={
            "image": "sheet.jpg", "threshold": 0.9,
            "readings": {"azure": {"scientificName": {"value": "Rumex Acetosella",
                                                      "confidence": 0.94}}},
            "truth": {"scientificName": "Rumex acetosella"}, "agreed": []})
        assert r.status_code == 200 and r.json()["fields"] == 1
        # a review with nothing resolved is not a review
        assert client.post("/reviews", json={"image": "x.jpg", "truth": {}}).status_code == 400
        assert client.get("/reviews").json()["reviews_logged"] == 1
        row = json.loads(open(m.REVIEWS, encoding="utf-8").readline())
        assert row["truth"]["scientificName"] == "Rumex acetosella"
        assert row["readings"]["azure"]["scientificName"]["confidence"] == 0.94
        assert "ts" in row
    finally:
        m.REVIEWS = original


def test_fetch_is_capped_like_upload():
    big = b"\x00" * (m.MAX_UPLOAD_BYTES + 1024)

    def fake_get(url, timeout=None, stream=False):
        return _Resp(big)

    with patch.object(m, "requests") as rq:
        rq.get = fake_get
        resp = TestClient(m.app).post("/azure", params={"url": "http://example.invalid/big"})
    assert resp.status_code == 413, resp.text


def test_compare_reports_a_pipeline_that_returns_an_error_payload():
    def failing(path):
        return json.dumps({"_error": "Document AI is not configured"})

    def working(path):
        return json.dumps({"scientificName": "Rumex acetosella", "confidence": {}})

    with patch.dict(m.PIPELINES, {"azure": working, "google": failing}):
        r = TestClient(m.app).post("/compare?pipelines=azure,google",
                                   files={"file": ("s.jpg", _png(3), "image/jpeg")})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert res["azure"]["ok"]
    assert not res["google"]["ok"], "an _error payload was reported as a good read"
    assert "not configured" in res["google"]["error"]


def test_google_import_does_not_need_its_env_vars():
    import transcription.google_vision as gv
    saved = {v: os.environ.pop(v, None)
             for v in ("GOOGLE_PROJECT_ID", "GOOGLE_PROCESSOR_ID")}
    try:
        try:
            gv._docai_config()
            assert False, "expected a configuration error"
        except RuntimeError as e:
            assert "GOOGLE_PROJECT_ID" in str(e)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    test_concurrent_requests_do_not_share_a_temp_file()
    test_fetch_is_capped_like_upload()
    test_compare_reports_a_pipeline_that_returns_an_error_payload()
    test_google_import_does_not_need_its_env_vars()
    test_bad_image_is_rejected_without_leaking()
    test_every_pipeline_has_an_endpoint()
    test_upload_works_on_every_pipeline()
    test_upload_rejects_bad_input()
    test_compare_runs_several_pipelines_on_one_upload()
    test_original_routes_unchanged()
    test_ceilings_are_probabilities()
    test_review_logging()
    print("ok")
