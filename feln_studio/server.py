"""FELN Studio: one local playground over every North Sea FELN backend.

  uv run --no-sync python -m feln_studio.server --start
Then open http://127.0.0.1:8766/. Backends that cannot initialise are skipped with a warning;
`--start` launches llama-server / mlx_lm.server when nothing healthy answers on their URLs.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from .backends import ROOT, Backend, LlamaBackend, MlxBackend, RagBackend
from .catalog import Gold, Schema, judge

STATIC = Path(__file__).with_name("static")
SIBLINGS = ROOT.parent
log = logging.getLogger(__name__)


class Studio:
    def __init__(self, backends: list[Backend], schema: Schema, gold: Gold):
        if not backends:
            raise ValueError("Register at least one backend")
        self.backends = {b.key: b for b in backends}
        self.schema, self.gold = schema, gold

    def config(self) -> dict:
        return {
            "backends": [
                {
                    "key": b.key,
                    "label": b.label,
                    "family": b.family,
                    "prompt": b.prompt,
                    "prompt_label": b.prompt_label,
                    "info": b.info,
                    "decoding": b.decoding,
                    "retrieves": b.retrieves,
                    "healthy": b.health(),
                }
                for b in self.backends.values()
            ],
            "questions": self.gold.questions(),
            "catalog": self.schema.path.name,
        }

    def retrieve(self, key: str, query: str) -> dict:
        return {"examples": self.backends[key].retrieve(query)}

    def generate(self, key: str, query: str, prompt: str | None, ids: list[int] | None) -> dict:
        backend = self.backends[key]
        started = time.monotonic()
        with backend.lock:
            try:
                result = backend.generate(query, backend.prompt if prompt is None else prompt, ids)
                feln = self.schema.compile(result["parsed"])
            except (ValueError, OSError) as exc:  # OSError: local model server unreachable
                return {"valid": False, "error": str(exc), "seconds": time.monotonic() - started}
        expected, exact = judge(self.schema, feln, self.gold.lookup(query))
        return {
            "valid": True,
            "feln": feln,
            "raw": result["raw"],
            "seconds": time.monotonic() - started,
            "first_token_seconds": result.get("first_token_seconds"),
            "prompt_tokens": result.get("prompt_tokens"),
            "generated_tokens": result.get("generated_tokens"),
            "examples": result.get("examples"),
            "expected": expected,
            "exact": exact,
        }


def handler_for(app: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Questions and model output stay out of access logs.

        def send(self, status: int, content: bytes, mime: str):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def json(self, status: int, payload: dict):
            self.send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

        def local_request(self) -> bool:
            port = cast(ThreadingHTTPServer, self.server).server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin") in {
                None,
                *(f"http://{host}" for host in hosts),
            }

        def do_GET(self):
            if not self.local_request():
                return self.json(403, {"error": "Local requests only"})
            if self.path == "/api/config":
                return self.json(200, app.config())
            files = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if self.path not in files:
                return self.json(404, {"error": "Not found"})
            name, mime = files[self.path]
            self.send(200, (STATIC / name).read_bytes(), mime + "; charset=utf-8")

        def do_POST(self):
            if not self.local_request():
                return self.json(403, {"error": "Local requests only"})
            if self.path not in {"/api/generate", "/api/retrieve"}:
                return self.json(404, {"error": "Not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.json(415, {"error": "Expected application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 200_000:
                    raise ValueError("Request must be between 1 and 200,000 bytes")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")  # noqa: TRY004 - client error
                query, key = data.get("query"), data.get("backend")
                prompt, ids = data.get("prompt"), data.get("ids")
                if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                    raise ValueError("Enter a question of 1–2,000 characters")
                if not isinstance(key, str) or key not in app.backends:
                    raise ValueError("Choose one of the registered backends")
                backend = app.backends[key]
                if self.path == "/api/retrieve" and not backend.retrieves:
                    raise ValueError("This backend does not retrieve examples")
                if prompt is not None and (
                    not isinstance(prompt, str) or not 0 < len(prompt) <= 100_000
                ):
                    raise ValueError("Prompt must be 1–100,000 characters")
                if ids is not None:
                    if not backend.retrieves:
                        raise ValueError("This backend does not take example IDs")
                    count = len(cast(RagBackend, backend).index.examples)
                    if (
                        not isinstance(ids, list)
                        or len(ids) != 5
                        or len(set(ids)) != 5
                        or any(type(i) is not int or not 0 <= i < count for i in ids)
                    ):
                        raise ValueError("Select five distinct example IDs from the corpus")
            except (ValueError, UnicodeDecodeError) as exc:
                message = "Invalid JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
                return self.json(400, {"error": message})
            try:
                if self.path == "/api/retrieve":
                    result = app.retrieve(key, query.strip())
                else:
                    result = app.generate(key, query.strip(), prompt, ids)
            except Exception:
                log.exception("%s failed", self.path)
                return self.json(502, {"error": "Backend request failed; inspect the terminal."})
            self.json(200, result)

    return Handler


def launch(command: list[str], log_path: Path, healthy, wait_seconds: int = 180) -> None:
    """Start a local model server unless one already answers, and stop it on exit."""
    if healthy():
        return
    out = log_path.open("ab")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=out)
    atexit.register(process.terminate)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(1)
        if healthy():
            return
        if process.poll() is not None:
            raise RuntimeError(f"{command[0]} exited; inspect {log_path}")
    raise TimeoutError(f"{command[0]} did not become healthy in {wait_seconds}s")


def build(args: argparse.Namespace) -> Studio:
    backends: list[Backend] = []
    wanted = args.backends.split(",")
    if "lora" in wanted:
        try:
            lora = LlamaBackend("lora", args.lora_label, args.lora_bundle, args.lora_url)
            if args.start:
                launch(
                    [
                        "llama-server",
                        "-m",
                        str(args.lora_gguf),
                        "-c",
                        "2048",
                        "-np",
                        "1",
                        "-ngl",
                        "all",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        args.lora_url.rsplit(":", 1)[1],
                    ],
                    args.lora_gguf.with_suffix(".server.log"),
                    lora.health,
                )
            backends.append(lora)
        except Exception as exc:  # noqa: BLE001 - skip this backend, keep the rest
            log.warning("lora backend skipped: %s", exc)
    if "liquid" in wanted:
        try:
            liquid = MlxBackend(
                "liquid",
                args.liquid_label,
                args.liquid_system,
                args.liquid_adapter,
                args.liquid_url,
            )
            if args.start:
                launch(
                    [
                        os.environ.get("MLX_SERVER", "mlx_lm.server"),
                        "--model",
                        str(args.liquid_model),
                        "--adapter-path",
                        str(args.liquid_adapter),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        args.liquid_url.rsplit(":", 1)[1],
                    ],
                    args.liquid_adapter / "server.log",
                    liquid.health,
                )
            backends.append(liquid)
        except Exception as exc:  # noqa: BLE001 - skip this backend, keep the rest
            log.warning("liquid backend skipped: %s", exc)
    if "rag" in wanted:
        if not args.rag_model:
            log.warning("rag backend skipped: set LLM_MODEL_NAME or --rag-model")
        else:
            try:
                backends.append(
                    RagBackend(
                        "rag",
                        args.rag_label,
                        args.data / "FELN.json",
                        args.data / "Layers.json",
                        args.rag_cache,
                        args.rag_encoder,
                        args.rag_model,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - skip this backend, keep the rest
                log.warning("rag backend skipped: %s", exc)
    for item in args.llama or []:
        label, bundle, url = item.split("=", 2)
        try:
            backends.append(LlamaBackend(f"llama{len(backends)}", label, Path(bundle), url))
        except Exception as exc:  # noqa: BLE001 - skip this backend, keep the rest
            log.warning("%s skipped: %s", label, exc)
    gold = args.gold or [args.data / "FELN.json"]
    return Studio(backends, Schema(args.data / "Layers.json"), Gold(gold))


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--backends", default="lora,liquid,rag", help="comma list; default all")
    parser.add_argument("--start", action="store_true", help="launch local model servers")
    parser.add_argument(
        "--data",
        type=Path,
        default=SIBLINGS / "feln-rag/feln_rag/data/NorthSea",
        help="folder with Layers.json (catalog) and FELN.json (gold)",
    )
    parser.add_argument(
        "--gold",
        action="append",
        type=Path,
        help="question/FELN records to judge against (repeatable); default DATA/FELN.json",
    )
    parser.add_argument(
        "--llama",
        action="append",
        metavar="LABEL=BUNDLE=URL",
        help="another GGUF bundle (inference_config.json) served by llama-server at URL",
    )
    lora = parser.add_argument_group("lora (feln-lora GGUF via llama-server)")
    lora.add_argument("--lora-label", default="Nemotron-3-Nano-4B · LoRA · Q8_0")
    lora.add_argument(
        "--lora-bundle", type=Path, default=SIBLINGS / "feln-lora/runs/nemotron-mac-20260915/merged"
    )
    lora.add_argument(
        "--lora-gguf",
        type=Path,
        default=SIBLINGS
        / "feln-lora/runs/nemotron-mac-20260915/gguf/nemotron-4b-step443-q8_0.gguf",
    )
    lora.add_argument("--lora-url", default="http://127.0.0.1:8092")
    liquid = parser.add_argument_group("liquid (feln-liquid MLX adapter via mlx_lm.server)")
    liquid.add_argument("--liquid-label", default="LFM2.5-1.2B · MLX LoRA")
    liquid.add_argument(
        "--liquid-model", type=Path, default=SIBLINGS / "feln-liquid/models/LFM2.5-1.2B-Instruct"
    )
    liquid.add_argument(
        "--liquid-adapter",
        type=Path,
        default=SIBLINGS / "feln-liquid/artifacts/northsea-normalized-20260916/best",
    )
    liquid.add_argument(
        "--liquid-system",
        type=Path,
        default=SIBLINGS / "feln-liquid/artifacts/northsea-normalized-20260916/system.txt",
    )
    liquid.add_argument("--liquid-url", default="http://127.0.0.1:8093")
    rag = parser.add_argument_group("rag (feln-rag few-shot via litellm)")
    rag.add_argument("--rag-label", default="RAG · few-shot")
    rag.add_argument("--rag-model", default=os.environ.get("LLM_MODEL_NAME"))
    rag.add_argument("--rag-encoder", default="local:multi-qa-mpnet-base-dot-v1")
    rag.add_argument("--rag-cache", type=Path, default=ROOT / "indices/studio")
    args = parser.parse_args()
    app = build(args)
    # A plain kill must still run atexit, which stops the servers we started.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app)) as server:
        print(f"FELN Studio → http://127.0.0.1:{server.server_port}", flush=True)
        for b in app.backends.values():
            print(f"  {b.key:7} {b.label}: {b.info}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
