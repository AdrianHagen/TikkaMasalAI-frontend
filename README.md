TikkaMasalAI Frontend
======================

Simple Streamlit UI to classify a dish from an image and, after prediction, ask a local LLM how to prepare that dish.

Features
--------
- Image upload (JPG/PNG/WEBP)
- Calls prediction service at `/predict`
- Extracts the predicted dish label from flexible JSON shapes
- "Get recipe from LLM" button to query `/llm/generate` for a step-by-step recipe
- Explainability: calls `/predict/explain` and overlays an attention heatmap on the image
- Raw JSON expanders for debugging both responses

Configure endpoints
-------------------
- Prediction API URL (default): `http://127.0.0.1:8000/predict`
- LLM URL (default): `http://127.0.0.1:8000/llm/generate`
 - Explain URL (default): `http://127.0.0.1:8000/predict/explain`

You can override these in two ways:
1) Streamlit secrets (recommended for deployments)
	- `.streamlit/secrets.toml`:
     
	  [secrets]
	  api_url = "http://localhost:8000/predict"
	  llm_url = "http://localhost:8000/llm/generate"
	explain_url = "http://localhost:8000/predict/explain"

2) Query params (quick local testing)
	- `?api_url=http://localhost:8000/predict&llm_url=http://localhost:8000/llm/generate`
	- `&explain_url=http://localhost:8000/predict/explain`

How to run (local)
------------------
1) Ensure the backend is running on the configured URLs.
2) In this folder, run Streamlit:

	streamlit run main.py

Notes
-----
- The LLM request body is sent using a flexible fallback strategy to support different server schemas: it tries `{ "prompt": ... }`, `{ "text": ... }`, `{ "input": ... }`, and a chat-like `{ "messages": [{"role": "user", "content": ...}] }`.
- The response text is extracted from common fields like `text`, `response`, `output`, `content`, or OpenAI-like `choices[0].message.content`.
- The explain endpoint may return a base64 overlay image or a numeric heatmap. If a heatmap is returned, the app overlays a red transparency mask (configurable opacity) on the original image.
