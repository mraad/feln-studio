# feln-studio — one SPA over every FELN backend + map

## Findings (2026-09-16)

No map app exists anywhere in the feln siblings. Closest pieces:
- `feln-lora/src/spatial_query.py` — FELN → parameterized read-only DuckDB → GeoJSON FeatureCollection
  (already emits geometry). Only consumer is `mcp_server.execute_feln`, which strips geometry.
- `feln/feln/sql.py` `FELNToDuckDB` — SQL compiler (`st_as='ST_AsGeoJSON'`), no execution.
- `feln-webgpu/web/{db.js,feln_sql.js}` — duckdb-wasm + parquet + JS SQL port; step 8 (index.html) never done.
- No leaflet/maplibre/openlayers in any feln project.

## Plan

- [x] 1. Scaffold `feln_studio/` (uv, py3.13): deps `feln`, `feln-rag` (path), `duckdb`, `sqlglot`, `python-dotenv`.
- [x] 2. `backends.py` — one protocol `generate(query, prompt|None) -> {feln, raw, timings}`:
      `rag` (import feln_rag: retrieve 5 + litellm), `lora` (llama-server /completion, GGUF, JSON grammar),
      `liquid` (mlx_lm.server /v1/chat/completions with adapter, system.txt). Each declares label, default prompt, health.
- [x] 3. `catalog.py` — copy `Schema` (validate/compile/context) from feln-lora; gold from NorthSea FELN.json, shared by all backends → strict verdict everywhere.
- [ ] 4. (deferred with map) `execute.py` — copy `SpatialQuery` from feln-lora; `/api/execute` → GeoJSON + SQL + count.
- [x] 5. `server.py` — one handler (merge of the two existing): `/api/config`, `/api/generate`, `/api/retrieve`, `/api/execute`, static. Same Host/Origin/CSP guards. Logs exceptions server-side.
- [x] 6. SPA: backend switcher, composer, highlighted FELN + raw toggle, verdict vs gold, evidence strip (examples | gold). Light + dark.
- [ ] 6b. Map panel: Leaflet (vendored, Carto Positron tiles), `/api/execute`, row table, SQL reveal.
- [x] 7. Tests: server contract with fake backends; executor on temp DuckDB; ruff + pyright.
- [x] 8. README + `--backends` CLI flags; run end-to-end against llama-server + mlx_lm.server + RAG.

## Deferred
- webgpu backend: in-browser transformers.js + duckdb-wasm. Different runtime (client-side, 300 MB model). Add as 4th tab once server-side three work.
- feln-lora `Schema`/`SpatialQuery` duplicated here (feln-lora is not an installable package). Make feln-lora installable and import instead when you're ready to touch it.

## Notes (2026-09-16)
- mlx_lm 0.31.3 server drops `--adapter-path` for `default_model` (resolves the model alias before
  the adapter lookup). `MlxBackend` names the adapter per request; keep that until mlx_lm fixes it.
- Gold = feln-rag's public `FELN.json` (`LIKE`); feln-lora was trained on normalized records
  (`ILIKE`), so lora/rag answers to those questions judge as mismatch. Point `--data` at a
  normalized bundle to judge lora on its own gold.
