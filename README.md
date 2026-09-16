# FELN Studio

One local playground over every North Sea natural-language → FELN backend. Same question,
same catalog, same strict judge; pick the backend and inspect exactly what it decodes.

| Backend | Sibling | Runtime | Prompt shown in the UI |
|---|---|---|---|
| `lora` | `../feln-lora` | llama-server, GGUF, JSON-schema grammar | the trained prompt prefix |
| `liquid` | `../feln-liquid` | `mlx_lm.server`, LFM2.5-1.2B + LoRA adapter | the run's `system.txt` |
| `rag` | `../feln-rag` | sentence-transformers retrieval + litellm | system prompt & catalog |

Every answer is schema-compiled through the catalog (`Layers.json`) and, when the question is
a recorded one (`FELN.json`), judged with `FELN.same` against the compiled gold. The raw model
text stays one click away.

## Run

```bash
uv sync
cp .env.example .env            # LLM_MODEL_NAME for rag; MLX_SERVER for liquid
uv run --no-sync python -m feln_studio.server --start
```

Open <http://127.0.0.1:8766/>. `--start` launches `llama-server` (port 8092) and
`mlx_lm.server` (port 8093) when nothing healthy answers there, and stops them on exit.
Without `--start`, run them yourself:

```bash
llama-server -m ../feln-lora/runs/nemotron-mac-v2-20260916/lora/gguf/nemotron-4b-v2-lora-q8_0.gguf \
  -c 2048 -np 1 -ngl all --host 127.0.0.1 --port 8092
mlx_lm.server --model ../feln-liquid/models/LFM2.5-1.2B-Instruct \
  --adapter-path ../feln-liquid/artifacts/northsea-normalized-20260916/best --port 8093
```

`--backends lora,rag` limits the set; a backend whose artifacts are missing is skipped with a
warning. `--help` lists the per-backend paths, all defaulting to the sibling checkouts.
`--llama LABEL=BUNDLE=URL` registers any other served GGUF bundle, e.g. the QLoRA model next to
the LoRA one; `--gold FILE` judges against other records (feln-lora's `tests/challenge.json`):

```bash
llama-server -m ../feln-lora/runs/nemotron-mac-qlora-20260916/gguf/nemotron-4b-qlora-step443-q8_0.gguf \
  -c 2048 -np 1 -ngl all --host 127.0.0.1 --port 8094 &
uv run --no-sync python -m feln_studio.server --start \
  --llama "QLoRA NF4 · Q8_0=../feln-lora/runs/nemotron-mac-qlora-20260916/merged=http://127.0.0.1:8094" \
  --gold ../feln-lora/tests/challenge.json
```

## Checks

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync pyright
```

## Layout

```
feln_studio/backends.py   Backend contract + LlamaBackend, MlxBackend, RagBackend
feln_studio/catalog.py    Schema (validate/compile), Gold records, judge()
feln_studio/server.py     loopback HTTP: /api/config /api/retrieve /api/generate, static
feln_studio/static/       vanilla JS SPA, light + dark
tests/                    server contract over fake backends
```

Map execution (FELN → DuckDB → GeoJSON on a map) is planned; see `tasks/todo.md`.
