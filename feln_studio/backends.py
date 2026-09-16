"""One contract over every FELN backend.

`generate(query, prompt, ids)` returns the raw model text and the JSON object parsed from
it; the server schema-compiles and judges. Backends never compare, never compile.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|startoftext|>")


def shown(path: Path) -> str:
    """Workspace-relative when possible: the UI must not leak the home directory."""
    root = ROOT.parent
    return str(path.relative_to(root)) if path.is_relative_to(root) else path.name


def get(url: str, timeout: float = 3) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def post(url: str, payload: dict, timeout: float = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_json(raw: str) -> dict:
    """The one JSON object in a completion; fences and chatter around it are tolerated."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip())
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object in the model output")
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")  # noqa: TRY004 - model output, not a type bug
    return value


class Backend:
    key: str
    label: str
    family: str  # rag | llama | mlx
    prompt: str  # editable default; the UI sends it back verbatim or edited
    prompt_label: str
    info: str
    decoding: str
    retrieves = False

    def __init__(self) -> None:
        # ponytail: one request at a time per backend; local servers run a single slot.
        self.lock = Lock()

    def health(self) -> bool:
        return True

    def retrieve(self, query: str) -> list[dict]:
        raise NotImplementedError

    def generate(self, query: str, prompt: str, ids: list[int] | None = None) -> dict:
        raise NotImplementedError


class LlamaBackend(Backend):
    """feln-lora GGUF bundle behind llama-server: raw prompt, JSON-schema grammar."""

    family = "llama"
    prompt_label = "Trained prompt"
    decoding = "Greedy · JSON-schema grammar · llama.cpp"

    def __init__(self, key: str, label: str, bundle: Path, url: str):
        super().__init__()
        self.key, self.label, self.url = key, label, url.rstrip("/")
        self.config = json.loads((bundle / "inference_config.json").read_text())
        self.prompt = self.config["prompt_prefix"]
        self.info = f"{shown(bundle)} → {self.url}"

    def health(self) -> bool:
        try:
            return b'"ok"' in get(self.url + "/health")
        except (urllib.error.URLError, OSError):
            return False

    def generate(self, query: str, prompt: str, ids: list[int] | None = None) -> dict:
        if any(token in query for token in CONTROL_TOKENS):
            raise ValueError("Question contains reserved model control tokens")
        result = post(
            self.url + "/completion",
            {
                "prompt": prompt + query + self.config["prompt_suffix"],
                "n_predict": self.config["max_new_tokens"],
                "temperature": 0,
                "json_schema": self.config["json_schema"],
                "cache_prompt": True,
            },
        )
        raw = result.get("content", "")
        if result.get("truncated") or result.get("stopped_limit"):
            raise ValueError(f"Model reached its output limit\nRaw output: {raw}")
        timings = result.get("timings", {})
        return {
            "raw": raw,
            "parsed": parse_json(raw),
            "prompt_tokens": timings.get("prompt_n"),
            "generated_tokens": timings.get("predicted_n"),
            "first_token_seconds": timings.get("prompt_ms", 0) / 1000,
        }


class MlxBackend(Backend):
    """feln-liquid MLX adapter behind mlx_lm.server: chat template, system prompt from the run."""

    family = "mlx"
    prompt_label = "System prompt"
    decoding = "Greedy · chat template · MLX"

    def __init__(
        self, key: str, label: str, system: Path, adapter: Path, url: str, max_tokens: int = 384
    ):
        super().__init__()
        self.key, self.label, self.url = key, label, url.rstrip("/")
        self.prompt = system.read_text().rstrip("\n")
        self.adapter, self.max_tokens = adapter, max_tokens
        self.info = f"{shown(adapter)} → {self.url}"

    def health(self) -> bool:
        try:
            return b"data" in get(self.url + "/v1/models")
        except (urllib.error.URLError, OSError):
            return False

    def generate(self, query: str, prompt: str, ids: list[int] | None = None) -> dict:
        result = post(
            self.url + "/v1/chat/completions",
            {
                "model": "default_model",
                # mlx_lm 0.31 resolves default_model before the adapter lookup and drops
                # --adapter-path; naming the adapter per request loads it for sure.
                "adapters": str(self.adapter),
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            },
        )
        choice = result["choices"][0]
        raw = choice["message"]["content"] or ""
        if choice.get("finish_reason") == "length":
            raise ValueError(f"Model reached its output limit\nRaw output: {raw}")
        usage = result.get("usage", {})
        return {
            "raw": raw,
            "parsed": parse_json(raw),
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "first_token_seconds": None,
        }


class RagBackend(Backend):
    """feln-rag: five nearest worked examples as chat turns, litellm completion."""

    family = "rag"
    prompt_label = "System prompt & catalog"
    decoding = "Top 5 · cosine · few-shot chat"
    retrieves = True

    def __init__(
        self, key: str, label: str, feln: Path, layers: Path, cache: Path, encoder: str, model: str
    ):
        super().__init__()
        from feln_rag.index import Encoder, build_or_load, load_examples
        from feln_rag.rag import system_prompt
        from layers_json.layers import Layers

        self.key, self.label, self.model = key, label, model
        self.prompt = system_prompt(Layers.load(str(layers)))
        self.encoder = Encoder(encoder)
        self.index = build_or_load(load_examples(feln), self.encoder, cache)
        if len(self.index.examples) < 5:
            raise ValueError("RAG needs at least five examples")
        self.info = (
            f"{model} · {encoder.removeprefix('local:')} · {len(self.index.examples)} examples"
        )

    def retrieve(self, query: str) -> list[dict]:
        hits = self.index.search(self.encoder.encode([query])[0], 5)
        return [{"id": i, "score": score, **self.index.examples[i]} for score, i in hits]

    def generate(self, query: str, prompt: str, ids: list[int] | None = None) -> dict:
        from feln_rag.rag import generate

        examples = (
            self.retrieve(query)
            if ids is None
            else [{"id": i, **self.index.examples[i]} for i in ids]
        )
        shots = [{"text": e["text"], "meta": e["meta"]} for e in examples]
        _, raw, _ = generate(query, prompt, lambda q, k: shots, model=self.model)
        return {"raw": raw, "parsed": parse_json(raw), "examples": examples}
