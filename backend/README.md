# SatQuery

## Groq supervisor configuration

The query classifier uses Groq's OpenAI-compatible API:

```env
GROQ_API_KEY="gsk_your_key_here"
GROQ_URL="https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL="openai/gpt-oss-20b"
```

Keep the API key in the ignored `.env` file and never commit it.

## Hugging Face authentication

Set the token in your shell before starting the backend. Do not commit the token.

```bash
export HF_TOKEN_BASE="hf_base_model_token"
export HF_TOKEN1="hf_single_image_token"
export HF_TOKEN2="hf_grounding_token"
export HF_TOKEN3="hf_change_analysis_token"
export HF_TOKEN4="hf_optical_sar_token"
export HF_XET_HIGH_PERFORMANCE=1
python main.py
```

`HF_XET_HIGH_PERFORMANCE` replaces the deprecated `HF_HUB_ENABLE_HF_TRANSFER` setting.

The tokens map to the repositories in `model_loader.py`: `HF_TOKEN1` to
`HF_MODEL_REPO1`, through `HF_TOKEN4` to `HF_MODEL_REPO4`. Keep tokens in your
shell or an ignored `.env` file and never commit them.