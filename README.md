# llm-vision-proxy

Transparent vision gateway that gives **text-only LLMs** (DeepSeek, GLM, …) the
ability to understand images, by describing images via the **Doubao** vision model
and forwarding text-only requests to the real backend.

```
Claude Code  ──POST /v1/messages (with image blocks)──►  llm-vision-proxy (127.0.0.1:8417)
   │                                                          ├─ route by `model` → real backend
   │                                                          ├─ describe each image via Doubao (Ark /api/v3/responses)
   │                                                          └─ replace image blocks with text blocks
   │                                                          │  (text-only request)
   ▼                                                          ▼
   ◄── streamed response (unchanged) ─────────────────  deepseek / glm / …
```

Claude Code points at the proxy; the proxy points at the real backend. Pasted images
just work — the text model receives a natural-language description instead of raw
pixels.

## Why

DeepSeek/GLM served over Anthropic-compatible endpoints silently drop images (the
model sees `[Unsupported Image]` and is blind). This proxy fixes that by converting
each image to a text description before the request reaches the backend.

## Configure

```bash
cp config.example.yaml config.yaml
chmod 600 config.yaml
# edit config.yaml: fill doubao.api_key + each upstream's api_key
```

`config.yaml` is gitignored and is the single source of truth for upstream
credentials. Keys are **never** committed.

## Run

```bash
.venv/bin/uvicorn vision_proxy.app:create_app --factory \
  --host 127.0.0.1 --port 8417
```

Or via systemd user service (see `deploy/llm-vision-proxy.service`):
```bash
systemctl --user enable --now llm-vision-proxy
```

## Point Claude Code at it

Use the helper (rewrites `~/.claude/settings.json` env, non-invasive):
```bash
scripts/vision-use deepseek-v4-flash   # or glm-latest, LongCat-2.0, …
```
This sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8417` and the chosen model. Switching
via `cc-switch use` overrides this (disabling vision for that session); re-run
`vision-use <model>` to restore.

## Endpoints

- `POST /v1/messages` — Anthropic Messages API (streaming passthrough)
- `POST /v1/messages/count_tokens` — token counting (images stripped first)
- `GET /v1/models` — configured upstreams
- `GET /healthz` — service + cache status

## Tests

```bash
.venv/bin/pytest
```

## Backend status

The proxy routes by `model` to a real text-only backend and is format-agnostic about
backend health. Verified end-to-end (image -> Doubao description -> backend answers
correctly) on this server:

| Upstream model      | Backend                  | Status |
|---------------------|--------------------------|--------|
| `deepseek-v4-flash` | api.deepseek.com/anthropic | ✅ works |
| `LongCat-2.0`       | api.longcat.chat/anthropic | ✅ works |
| `MiniMax-M3`        | api.minimaxi.com/anthropic | ✅ works |
| `glm-latest`        | ark …/api/plan           | ⚠️ routed, but Ark **AgentPlan subscription expired** - renew in Ark console, then works with no code change |
| `mimo-v2.5-pro`     | xiaomimimo.com/anthropic | ⚠️ routed, but cc-switch token **invalid (401)** - refresh the mimo token |

Doubao vision model (`doubao-seed-2-0-lite-260428` via Ark `/api/v3/responses`) is
verified working as the image describer.

To add/refresh a backend's credentials, edit `config.yaml` (`upstreams:`) and
`systemctl --user restart llm-vision-proxy`, or re-run
`scripts/extract-upstreams.py` after updating the cc-switch provider.

## Security

- Listens on `127.0.0.1` only.
- `config.yaml` mode 600, gitignored; real keys live only here.
- Logs contain only metadata (model, image counts, latency, errors) — never image
  content or keys.
