# Barrikade Open Day Demo

This demo showcases live attack-vs-defense behavior:

- WITH BARRIKADE: layered pipeline verdict and explanation (always runs on Analyze)
- WITHOUT SECURITY: optional direct model output from the local Layer E model (runs only when clicked)

## Run

From repository root:

```bash
source .venv/bin/activate
streamlit run demo/streamlit_app.py
```

## Runtime requirement

The demo uses the local Qwen3Guard bundle for baseline and Layer E execution.

If required artifacts are missing, point the demo at `core/models/layer_e/qwen3guard-barrikade` or set `BARRIKADE_LAYER_E_MODEL_DIR`.

## Open Day flow

1. Click one of the attack presets.
2. Click Analyze to run the prompt through Barrikade.
3. Optionally click Run Base LLM (without security) for comparison.
4. Open layer tabs to explain where and why the verdict was made.
