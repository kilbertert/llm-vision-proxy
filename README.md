# llm-vision-proxy

A **thin vision shim** that gives text-only LLMs image understanding by describing
images via the **Doubao** vision model, then forwarding the text-only request to a
downstream gateway (**cliproxyapi**) which owns all model routing.

```
Codex CLI / Claude Code  (any model: gpt-5.6-sol, deepseek-v4-flash-ga-260731, deepseek-v4-flash, glm-latest, …)
   │  POST /v1/messages | /v1/chat/completions | /v1/responses  (with image content)
   ▼
llm-vision-proxy (127.0.0.1:8417)   ← thin shim: strip images -> Doubao description -> text-only
   │  (text-only request, unchanged otherwise; same path)
   ▼
cliproxyapi (127.0.0.1:8317)        ← SINGLE model-supply manager: routes by `model`
   ├─ gpt-5.6-sol                  -> codex-api-key
   ├─ deepseek-v4-flash-ga-260731  -> openai-compatibility (Ark)
   ├─ deepseek-v4-flash            -> openai-compatibility (api.deepseek.com)
   └─ glm-latest                   -> claude-api-key (Ark)
```

**Model supply is managed in exactly one place: cliproxyapi's `config.yaml`.** This
proxy does no model routing and holds no upstream credentials (only the Doubao key
and the cliproxyapi forward key). The response/stream is pure passthrough - no
protocol conversion, no model logic.

## Why

DeepSeek/GLM served over OpenAI/Anthropic-compatible endpoints silently drop images
(the model sees `[Unsupported Image]` and is blind). This shim fixes that by
converting each image to a text description before the request reaches the backend.

## Configure

```bash
cp config.example.yaml config.yaml
chmod 600 config.yaml
# edit config.yaml: fill doubao.api_key + forward.api_key (cliproxyapi's key)
```

`config.yaml` is gitignored. The cliproxyapi forward key is the same one Codex/Claude
Code present to cliproxyapi (e.g. `CLIPROXYAPI_KEY`).

To add a new model: add it to **cliproxyapi's** `config.yaml` (openai-compatibility /
claude-api-key / codex-api-key). Nothing changes here.

## Run

```bash
.venv/bin/uvicorn vision_proxy.app:create_app --factory --host 127.0.0.1 --port 8417
# or: systemctl --user enable --now llm-vision-proxy
```

## Endpoints

- `POST /v1/messages`, `POST /v1/messages/count_tokens` - Anthropic Messages (Claude Code)
- `POST /v1/chat/completions` - OpenAI Chat Completions
- `POST /v1/responses` - OpenAI Responses (Codex CLI)
- `GET /v1/models` - proxied to cliproxyapi (its catalog)
- `GET /healthz` - status + forward target + cache

Each endpoint strips images in its own format (Anthropic content blocks / OpenAI
`image_url` parts / Responses `input_image` parts), then forwards to cliproxyapi.

## Codex CLI (fast model switching)

`~/.codex/config.toml` uses a single provider `visionproxy` (the shim, `wire_api =
"responses"`). Switching models is just `model` + `model_reasoning_effort`:

```bash
scripts/codex-use daily   # deepseek-v4-flash-ga-260731, effort=high (vision-enabled)
scripts/codex-use hard    # gpt-5.6-sol, effort=max
```

Takes effect on the next `codex` invocation. Use this instead of `cc-switch use ...
-a codex` (its codex profiles are stale and would overwrite the shim provider).

Note: `deepseek-v4-flash-ga` at `model_reasoning_effort = "max"` is slow and
occasionally stalls (Codex stream-parsing quirks with this custom model), so daily
defaults to `high`. Codex emits non-fatal `"... without active item"` warnings; text
+ image responses work.

## Claude Code

```bash
scripts/vision-use                      # list models from cliproxyapi catalog
scripts/vision-use deepseek-v4-flash-ga-260731   # route Claude Code through the shim
scripts/vision-use off                  # restore original settings
```

## Tests

```bash
.venv/bin/pytest   # 52 tests
```

## Security

- Listens on `127.0.0.1` only.
- `config.yaml` mode 600, gitignored; holds only the Doubao key + cliproxyapi forward
  key. Backend credentials live in cliproxyapi, not here.
- Logs contain only metadata (format, model, image counts, latency, errors) - never
  image content or keys.
