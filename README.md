# SatQuery

SatQuery is a satellite-image question-answering application. Upload one or
two satellite images, ask a natural-language question, and receive analysis
from a Qwen2.5-VL model with task-specific adapters for visual question
answering, captioning, grounding, change analysis, and optical/SAR analysis.

## Video Demonstration

> Add the project walkthrough or demo video here.

<!--
Replace the placeholder below with a YouTube, Loom, or local video link.
Example:


-->

**Demo video:**
[![Watch the SatQuery demonstration](https://img.youtube.com/vi/_3UY2hBNBjk/hqdefault.jpg)](https://youtu.be/_3UY2hBNBjk)


## Features

- Ask questions about a single satellite image in natural language.
- Count objects and answer visual questions with image evidence.
- Generate a concise description of a complete scene.
- Locate requested objects with normalized bounding boxes.
- Compare two observations for bi-temporal change analysis.
- Analyze complementary optical and SAR observations.
- Accept PNG, JPEG, and GeoTIFF inputs.
- Display analysis answers, task details, latency, and grounding results in a
  chat-style web interface.

## Architecture

```text
Browser
  |
  | Next.js / React frontend
  | POST multipart form data
  v
FastAPI backend (:8000)
  |
  +--> Validate and decode images
  +--> Route query using deterministic remote-sensing rules
  +--> Activate the matching preloaded adapter
  +--> Run Qwen2.5-VL specialist workflow
  +--> Return answer, tasks, detections, and execution trace
```

The backend loads the base model and adapters during import. This means model
initialization happens before Uvicorn accepts requests and can take several
minutes on the first run.

### Model and workflow mapping

| Request type | Class | Adapter | Typical use |
| --- | --- | --- | --- |
| Single-image VQA | `SINGLE_IMAGE` | `vqa` | Questions, counting, visual evidence |
| Grounding/captioning | `GROUNDING_CAPTIONING` | `captioning` | Scene descriptions and object locations |
| Two-date comparison | `CHANGE_ANALYSIS` | `change` | Before/after and temporal change questions |
| Two-image fusion | `OPTICAL_SAR` | `optical_sar` | Complementary optical and SAR observations |

Routing is currently performed by local keyword and image-count rules in
`backend/supervisor.py`. The `GROQ_*` variables remain available for future
supervisor integration, but the current request path does not make a Groq API
call.

## Repository Structure

```text
satquery/
├── backend/
│   ├── main.py              # FastAPI application and API endpoints
│   ├── supervisor.py        # Query classification and workflow routing
│   ├── vlm.py               # Adapter selection and VLM inference
│   ├── model_loader.py      # Base model and adapter loading
│   ├── image_validation.py  # Image validation and GeoTIFF conversion
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Backend-specific notes
└── frontend/
    ├── app/                 # Next.js app shell and global styles
    ├── components/          # Chat UI and SatQuery state
    ├── data/                # Globe visualization data
    ├── public/               # Static assets and textures
    ├── package.json         # Frontend scripts and dependencies
    └── next.config.mjs      # Next.js configuration
```

## Requirements

- Linux with a CUDA-capable GPU recommended.
- Python 3.10+ compatible with the installed PyTorch and Unsloth versions.
- Node.js and pnpm. The frontend declares `pnpm@12.3.4` as its package
  manager.
- Hugging Face access to the base model and adapter repositories.
- Enough disk space for model downloads and enough GPU memory for the loaded
  Qwen2.5-VL model and adapters. The default configuration uses 4-bit model
  loading.

The backend imports `unsloth`, `torch`, and the model loader immediately, so a
missing dependency or invalid Hugging Face credential prevents the API from
starting.

## Setup

### 1. Configure backend credentials

Create an ignored `backend/.env` file, or export the variables in your shell.
Do not commit tokens or API keys.

```env
HF_TOKEN_BASE=hf_your_base_model_token
HF_TOKEN1=hf_your_vqa_adapter_token
HF_TOKEN2=hf_your_grounding_adapter_token
HF_TOKEN3=hf_your_change_adapter_token
HF_TOKEN4=hf_your_optical_sar_adapter_token
```

If the adapter repositories use the same Hugging Face credential, setting only
`HF_TOKEN_BASE` is sufficient because it is used as the fallback token.

### 2. Install and start the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The API listens on `http://localhost:8000`. The server starts only after the
base model, four adapters, and the SAR components have loaded.

For a frontend running on another origin, configure the allowed origins:

```bash
export FRONTEND_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
```

### 3. Install and start the frontend

In a second terminal:

```bash
cd frontend
pnpm install
NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in a browser. The
frontend defaults to `http://localhost:8000` when `NEXT_PUBLIC_API_URL` is not
set.

## Configuration

### Backend variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN_BASE` | None | Token for the base model; fallback for adapter tokens |
| `HF_TOKEN1`-`HF_TOKEN4` | `HF_TOKEN_BASE` | Tokens for the four adapter repositories |
| `BASE_MODEL` | `unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit` | Base Hugging Face model |
| `HF_MODEL_REPO1` | VQA adapter repository | Single-image VQA adapter |
| `HF_MODEL_REPO2` | Grounding adapter repository | Captioning and grounding adapter |
| `HF_MODEL_REPO3` | Change adapter repository | Change-analysis adapter |
| `HF_MODEL_REPO4` | Optical/SAR adapter repository | Optical/SAR adapter |
| `MAX_NEW_TOKENS` | `256` | Maximum generated tokens |
| `FRONTEND_ORIGINS` | Local ports `3000` | Comma-separated CORS origins |
| `LOG_LEVEL` | `INFO` | Backend logging level |
| `GROQ_API_KEY` | None | Reserved for supervisor integration |
| `GROQ_URL` | Groq chat completions URL | Reserved supervisor endpoint |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Supervisor model configuration |

### Frontend variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API origin |

## API Reference

### `GET /health`

Checks that the model-backed API is available.

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "base_model": "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
  "supervisor": "openai/gpt-oss-20b",
  "device": "cuda:0"
}
```

### `GET /models`

Returns the configured base model and adapter registry.

### `POST /classify`

Accepts `multipart/form-data` with:

- `query`: required string
- `image_count`: required integer
- `modalities`: optional string, default `unknown`

Returns the selected classes, workflow names, and routing parameters without
running specialist inference.

### `POST /analyze`

Accepts `multipart/form-data` with:

- `query`: required string
- `image1`: required image file
- `image2`: optional second image file
- `modalities`: optional string, default `unknown`

Supported extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`. The API
supports one or two images only.

Example request:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "query=How many buildings are visible?" \
  -F "image1=@/path/to/satellite-image.png" \
  -F "modalities=unknown"
```

The response contains:

```json
{
  "request_id": "...",
  "answer": "...",
  "tasks": [
    {
      "task": "SINGLE_IMAGE",
      "result": {
        "answer": "...",
        "boxes": [],
        "caption": null,
        "evidence_image": null,
        "is_grounding": false,
        "latency_seconds": 1.23
      },
      "model": {}
    }
  ],
  "execution_trace": {
    "selected_tasks": ["SINGLE_IMAGE"],
    "workflow": ["visual_question_answering"],
    "total_latency_seconds": 2.34
  }
}
```

## Image Handling

- PNG, JPEG, and JPG files are converted to RGB.
- Single-band GeoTIFFs are normalized to grayscale RGB.
- GeoTIFFs with three or more bands use the first three bands as RGB.
- GeoTIFF channels are normalized using the 2nd and 98th percentiles.
- Original raster metadata is not preserved in the model input.

## Frontend Workflow

1. Attach one or two supported images in the composer.
2. Enter a natural-language question.
3. The client builds a `FormData` request and posts it to `/analyze`.
4. The backend selects a workflow, activates the matching adapter, and runs
   inference.
5. The chat thread renders the answer, task output, image previews, and
   grounding detections when available.

The current UI also includes dataset, tool, and recent-chat surfaces. These
are frontend scaffolding: chat history is held in React state and is not yet
persisted to a database or backed by chat API endpoints.

## Troubleshooting

### `TypeError: Failed to fetch`

Check that the backend process is running and that the frontend points to the
same origin:

```bash
curl http://localhost:8000/health
```

If the frontend and backend use different origins, set both:

```bash
NEXT_PUBLIC_API_URL=http://BACKEND_HOST:8000 pnpm dev
FRONTEND_ORIGINS=http://FRONTEND_HOST:3000
```

Restart the backend after changing its environment variables. CORS settings
are read when FastAPI starts.

### `ModuleNotFoundError: No module named 'fastapi'`

Start the backend with the virtual environment that received the dependency
installation:

```bash
cd backend
source venv/bin/activate
python main.py
```

### Hugging Face authentication or download errors

Confirm that the token can read every configured model repository and that the
repository variables point to valid model IDs. The first startup downloads
large model files; allow additional time and disk space for the cache.

### GPU or memory errors

The default model configuration expects a CUDA-capable environment. Check
PyTorch CUDA visibility, close other GPU workloads, and confirm that the
selected model and adapters fit available VRAM.

## Current Limitations

- Only one or two input images are supported per analysis request.
- `modalities` is accepted by the API but does not currently control routing.
- The optical/SAR route uses the loaded specialist adapter workflow; the
  separate fusion helper is not called directly by `/analyze`.
- Frontend chat history, authentication, datasets, and advanced tools are not
  connected to persistent backend services.
- Analysis failures are logged in the browser console and clear the displayed
  result instead of presenting a dedicated error panel.
- `next.config.mjs` currently sets `typescript.ignoreBuildErrors` to `true`.
  Treat a successful production build as insufficient evidence of complete
  type safety until the existing type errors are resolved.

## Development Notes

Useful local checks:

```bash
cd backend
python -m py_compile main.py

cd ../frontend
pnpm exec tsc --noEmit
pnpm build
```

The frontend typecheck currently contains an existing `loadChat` signature
mismatch in `components/satquery-context.tsx`; it is independent of the API
request path. Keep secrets in ignored environment files and avoid committing
model caches, virtual environments, API keys, or Hugging Face tokens.

## License

No license file is currently included in this repository. Add the appropriate
license before redistributing the project.