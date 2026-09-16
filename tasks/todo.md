# feln-studio — one SPA over every FELN backend + map

## Findings (2026-09-16)

No map app exists anywhere in the feln siblings. Closest pieces:
- `feln-lora/src/spatial_query.py` — FELN → parameterized read-only DuckDB → GeoJSON FeatureCollection
  (already emits geometry). Only consumer is `mcp_server.execute_feln`, which strips geometry.
- `feln/feln/sql.py` `FELNToDuckDB` — SQL compiler (`st_as='ST_AsGeoJSON'`), no execution.
- `feln-webgpu/web/{db.js,feln_sql.js}` — duckdb-wasm + parquet + JS SQL port; step 8 (index.html) never done.
- No leaflet/maplibre/openlayers in any feln project.

## Done (2026-09-16)

- [x] `feln_studio/` scaffold; deps `feln`, `feln-rag`, `layers-json` (sibling paths), `sqlglot`, `python-dotenv`.
- [x] `backends.py` — `generate(query, prompt, ids) -> {raw, ...}` + optional `retry(...)`;
      `lora` (llama-server, JSON grammar), `liquid` (mlx_lm.server, adapter per request, feln-liquid guard
      imported by path), `rag` (feln_rag retrieve 5 + litellm).
- [x] `catalog.py` — `Schema` (copied from feln-lora), `Gold`, `judge()`; server parses, compiles, judges.
- [x] `server.py` — `/api/config`, `/api/retrieve`, `/api/generate`, static; Host/Origin/CSP; `--start`
      launches and reaps model servers; `--llama`, `--gold`, `--liquid-guard`.
- [x] SPA — backend rail, composer, highlighted FELN + raw toggle, verdict vs gold, examples, clarification
      badge, attempts. Light + dark.
- [x] Tests (fake backends, retry, clarification, guard loader); ruff, pyright.
- [x] feln-lora and feln-rag pruned of their studios on `studio-moved` branches, 0.2.0.
- [x] Defaults on the v2 LoRA GGUF; liquid inherits `predict_feln.py` RULES / clarification / retry.

## Next

- [ ] Map panel: copy `SpatialQuery` from feln-lora → `/api/execute` (GeoJSON + SQL + count);
      Leaflet vendored, Carto Positron tiles; row table, SQL reveal.
- [ ] Browser click-through check (headless screenshots only so far; API path verified live).

## Deferred
- webgpu backend: in-browser transformers.js + duckdb-wasm. Different runtime (client-side, 300 MB model). Add as 4th tab once server-side three work.
- feln-lora `Schema`/`SpatialQuery` duplicated here (feln-lora is not an installable package). Make feln-lora installable and import instead when you're ready to touch it.

## Notes (2026-09-16)
- mlx_lm 0.31.3 server drops `--adapter-path` for `default_model` (resolves the model alias before
  the adapter lookup). `MlxBackend` names the adapter per request; keep that until mlx_lm fixes it.
- Gold = feln-rag's public `FELN.json` (`LIKE`); feln-lora was trained on normalized records
  (`ILIKE`), so lora/rag answers to those questions judge as mismatch. Point `--data` at a
  normalized bundle to judge lora on its own gold.
