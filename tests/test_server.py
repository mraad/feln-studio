"""Server contract over fake backends; no model servers required."""

import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from feln_studio import backends, catalog, server

LAYERS = {
    "layers": [
        {
            "name": "Wells",
            "columns": [
                {
                    "name": "content_type",
                    "dtype": "SmallInteger",
                    "keyval": {"2": "Gas"},
                    "values": ["2"],
                    "hints": [],
                }
            ],
        }
    ]
}
GOLD = [
    {
        "text": "Show  gas wells",
        "meta": {"layers": ["Wells"], "where": ["content_type = 2"], "relations": []},
    }
]


class Fake(backends.Backend):
    family, prompt_label, info, decoding = "fake", "Prompt", "fake → nowhere", "none"

    def __init__(self, key, retrieves=False):
        super().__init__()
        self.key, self.label, self.prompt, self.retrieves = key, key.title(), "P:", retrieves
        self.calls = []
        self.index = SimpleNamespace(
            examples=[{"text": f"q{i}", "meta": GOLD[0]["meta"]} for i in range(7)]
        )

    def retrieve(self, query):
        return [{"id": i, "score": 1 - i / 10, **self.index.examples[i]} for i in range(5)]

    def generate(self, query, prompt, ids=None):
        self.calls.append((query, prompt, ids))
        if query == "fail":
            raise ValueError("Model reached its output limit")
        if query == "boom":
            raise RuntimeError("secret-provider-detail")
        if query == "deep":
            raise backends.Clarification("Do you mean water depth?")
        if query in ("bad", "bad-twice"):
            return {"raw": '{"layers": ["Wells"], "where": ["nope = 1"], "relations": []}'}
        raw = '{"layers": ["wells"], "where": ["content_type = \'Gas\'"], "relations": []}'
        return {"raw": raw, "generated_tokens": 5}

    def retry(self, query, prompt, raw, error, ids):
        self.calls.append(("retry", error))
        if query == "bad-twice":
            return {"raw": "not json"}
        return {"raw": '{"layers": ["Wells"], "where": ["content_type = 2"], "relations": []}'}


@pytest.fixture
def studio(tmp_path: Path):
    (tmp_path / "Layers.json").write_text(json.dumps(LAYERS))
    (tmp_path / "FELN.json").write_text(json.dumps(GOLD))
    fakes: list[backends.Backend] = [Fake("lora"), Fake("rag", retrieves=True)]
    app = server.Studio(
        fakes, catalog.Schema(tmp_path / "Layers.json"), catalog.Gold([tmp_path / "FELN.json"])
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(app)) as srv:
        Thread(target=srv.serve_forever, daemon=True).start()

        def call(path, data=None, headers=None):
            connection = HTTPConnection("127.0.0.1", srv.server_port)
            connection.request(
                "GET" if data is None else "POST",
                path,
                None if data is None else json.dumps(data),
                {"Content-Type": "application/json", **(headers or {})},
            )
            response = connection.getresponse()
            body = response.read()
            return response.status, (
                json.loads(body)
                if response.getheader("Content-Type", "").startswith("application/json")
                else body.decode()
            )

        yield app, call
        srv.shutdown()


def test_config_and_static(studio):
    _, call = studio
    status, config = call("/api/config")
    assert status == 200
    assert [b["key"] for b in config["backends"]] == ["lora", "rag"]
    assert config["backends"][1]["retrieves"] is True
    assert config["questions"] == ["Show  gas wells"]  # original text, not the casefolded key
    assert "FELN Studio" in call("/")[1]
    assert call("/../.env")[0] == 404
    assert call("/nope")[0] == 404


def test_generate_judges_against_gold(studio):
    app, call = studio
    status, result = call("/api/generate", {"query": "show gas   wells", "backend": "lora"})
    assert status == 200 and result["valid"] and result["exact"] is True
    assert "CAST(2 AS SMALLINT)" in result["feln"]["where"][0]  # schema-compiled model output
    assert "CAST(2 AS SMALLINT)" in result["expected"]["where"][0]  # gold compiled the same way
    assert "'Gas'" in result["raw"]
    assert app.backends["lora"].calls[-1] == ("show gas   wells", "P:", None)

    status, result = call(
        "/api/generate", {"query": "Unknown question", "backend": "lora", "prompt": "X:"}
    )
    assert status == 200 and result["expected"] is None and result["exact"] is None
    assert app.backends["lora"].calls[-1][1] == "X:"


def test_generate_errors(studio):
    _, call = studio
    status, result = call("/api/generate", {"query": "fail", "backend": "lora"})
    assert (status, result["valid"]) == (200, False) and "output limit" in result["error"]
    status, result = call("/api/generate", {"query": "boom", "backend": "lora"})
    assert status == 502 and "secret-provider-detail" not in json.dumps(result)
    for body in (
        {"query": "x", "backend": "missing"},
        {"query": "", "backend": "lora"},
        {"query": "x", "backend": "lora", "prompt": ""},
        {"query": "x", "backend": "lora", "ids": [0, 1, 2, 3, 4]},
        {"query": "x", "backend": "rag", "ids": [0, 0, 1, 2, 3]},
        {"query": "x", "backend": "rag", "ids": [True, 1, 2, 3, 4]},
        {"query": "x", "backend": "rag", "ids": [0, 1, 2, 3, 99]},
        ["bad shape"],
    ):
        assert call("/api/generate", body)[0] == 400, body
    assert (
        call("/api/generate", {"query": "x", "backend": "lora"}, {"Origin": "http://evil"})[0]
        == 403
    )
    assert call("/api/config", headers={"Host": "example.com"})[0] == 403


def test_retrieve(studio):
    app, call = studio
    status, result = call("/api/retrieve", {"query": "wells", "backend": "rag"})
    assert status == 200 and len(result["examples"]) == 5
    assert call("/api/retrieve", {"query": "wells", "backend": "lora"})[0] == 400
    status, result = call(
        "/api/generate", {"query": "wells", "backend": "rag", "ids": [4, 3, 2, 1, 0]}
    )
    assert status == 200 and app.backends["rag"].calls[-1][2] == [4, 3, 2, 1, 0]


def test_parse_json_tolerates_fences_and_chatter():
    assert backends.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert backends.parse_json('Sure: {"a": {"b": [1]}} trailing') == {"a": {"b": [1]}}
    for bad in ("no json here", "[1, 2]", '{"a": '):
        with pytest.raises(ValueError):
            backends.parse_json(bad)


def test_retry_and_clarification(studio):
    app, call = studio
    status, result = call("/api/generate", {"query": "bad", "backend": "lora"})
    assert status == 200 and result["valid"] and result["attempts"] == 2
    assert "Unknown field Wells" in app.backends["lora"].calls[-1][1]
    status, result = call("/api/generate", {"query": "bad-twice", "backend": "lora"})
    assert status == 200 and not result["valid"] and "JSON" in result["error"]
    status, result = call("/api/generate", {"query": "deep", "backend": "lora"})
    assert status == 200 and not result["valid"] and result["clarification"]
    assert result["error"] == "Do you mean water depth?"
    # A backend without retry() surfaces the first validation error.
    app.backends["rag"].retry = lambda *a: None
    status, result = call(
        "/api/generate", {"query": "bad", "backend": "rag", "ids": [0, 1, 2, 3, 4]}
    )
    assert status == 200 and not result["valid"] and "Unknown field" in result["error"]


def test_load_guard_from_sibling(tmp_path):
    assert backends.load_guard(tmp_path / "missing.py") is None
    guard = tmp_path / "guard.py"
    guard.write_text(
        'RULES = "R"\n\ndef clarification(text):\n    return "ask" if "depth" in text else None\n'
    )
    module = backends.load_guard(guard)
    assert module is not None
    assert module.RULES == "R" and module.clarification("depth") == "ask"
