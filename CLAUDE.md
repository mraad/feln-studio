# Project guidance

FELN Studio: one vanilla-JS SPA and loopback HTTP server over every sibling FELN backend
(`../feln-lora` GGUF via llama-server, `../feln-liquid` MLX adapter via mlx_lm.server,
`../feln-rag` few-shot via litellm). README.md has commands; `tasks/todo.md` the roadmap.

- Backends implement `feln_studio.backends.Backend.generate(query, prompt, ids)` and return
  `{raw, parsed}` only. Compilation (`catalog.Schema`) and judging (`catalog.judge`, `FELN.same`)
  live in the server so every backend is scored identically. Never compare inside a backend.
- `catalog.Schema` is a copy of `feln-lora/src/feln_data.Schema`; feln-lora is not installable.
  Fix bugs in both until it is.
- Default data is `../feln-rag/feln_rag/data/NorthSea` (public). No personal filesystem paths in
  code or docs; `.env` (git-ignored) carries `LLM_MODEL_NAME` and `MLX_SERVER`.
- Server: loopback only, Host + Origin checks, strict request validation, generic 502 to the
  client with the traceback logged to the terminal. Static assets are an allowlist.
- SPA: no build step, no frontend dependencies, text nodes for model/catalog content, keyboard
  access and loading states preserved, light and dark via `prefers-color-scheme`.
- `uv sync`; `uv run --no-sync pytest -q`; `ruff check`, `ruff format --check`, `pyright` before done.
  Tests use fake backends; never require a model server.
