# SatQuery AI architecture notes

These notes describe the code in this repository. The six-page SIH concept deck is available at [`SatQuery_Seal_Team6.pdf`](SatQuery_Seal_Team6.pdf); implementation status is summarized in the root [README](../README.md).

## Request path

1. The Next.js client collects a natural-language query and one or two files, then posts multipart data to `/api/backend/analyze`.
2. Next.js rewrites that path to `SATQUERY_BACKEND_URL` (default `http://127.0.0.1:8000`).
3. FastAPI `/analyze` reads the full upload, checks its extension, decodes standard images or GeoTIFF data, and records dimensions/name.
4. `SATQUERY_USE_CLF_ROUTER=true` selects `clf_router.route_with_clf`. A checked-in checkpoint uses text/image features to predict specialist labels. Missing checkpoints, router errors, or disabling the classifier falls back to local keyword rules in `supervisor.route_query`.
5. `vlm.execute_workflow` dispatches VQA, grounding/captioning, change analysis, or optical–SAR fusion. The model loader loads Qwen2.5-VL plus four LoRA adapters and separately loads SARMAE and a projector for optical–SAR.
6. The response includes an answer, task outputs, and `execution_trace` with selected labels, workflow, model/router data, input metadata, and request latency. Grounding may also include parsed boxes and a data-URL image.

## Module map

| Path | Responsibility |
| --- | --- |
| `frontend/app/page.tsx` | Browser upload/chat state, request submit, answer and grounding rendering. |
| `frontend/next.config.ts` | Backend API rewrite. |
| `backend/main.py` | FastAPI endpoints, image request handling, route selection, response trace, local CLI. |
| `backend/image_validation.py` | Extension checks and PIL/GeoTIFF decoding. TIFF data is converted to a display image; geospatial transforms are not passed to VLM workflows. |
| `backend/clf_router.py` | Classifier load, label mapping, threshold use, and fallback. |
| `backend/supervisor.py` | Deterministic keyword routing. `call_groq_supervisor` currently returns `route_query`; no Groq HTTP call is made. |
| `backend/satquery_classifier.py` | Router feature encoder, multi-label classifier, training CLI, and inference wrapper. |
| `backend/model_loader.py` | Import-time base model, adapter, SARMAE, and projector loading. |
| `backend/vlm.py` | Input resizing, model generation, grounding-box parsing, task dispatch, and result construction. |

## API contract from source

| Route | Request | Response |
| --- | --- | --- |
| `GET /health` | None | Status, base model, router type/checkpoint, device. |
| `GET /models` | None | Base model and model registry. |
| `POST /classify` | Multipart query, image count, optional modalities | Classes, workflow, parameters, router. |
| `POST /analyze` | Multipart query, required image1, optional image2/modalities | Request ID, answer, task outputs, execution trace. |

The backend has no database schema, auth layer, or automated tests in the checked-in tree. Runtime behavior was not verified because model loading requires a configured ML environment.

## Configuration facts

- Next.js proxy target: `SATQUERY_BACKEND_URL`, default `http://127.0.0.1:8000`.
- Model IDs, classifier settings, token aliases, and generation settings: see the root README's Environment variables table.
- `GROQ_API_KEY` and `GROQ_URL` are read as settings, but the active routing function does not make a network request.
- `USE_4BIT` is read by the API module; `model_loader.py` hard-codes `load_in_4bit=True`.
