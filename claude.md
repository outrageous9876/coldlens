# ColdLens

AI label intelligence for cold chain & pharma. Upload a photo of a medicine/vaccine/food label → extract batch, expiry, storage temp → flag compliance issues.

## Stack
- Backend: Python 3.12, FastAPI, OpenCV, EasyOCR, Groq LLM, pydantic
- Frontend: Next.js (simple, one upload page)
- Later: openFDA + pgvector RAG (Sanofi upgrade)

## Pipeline
image → OpenCV preprocessing (grayscale, denoise, deskew, perspective correction, adaptive threshold)
→ EasyOCR → Groq LLM extracts JSON {product, batch_no, mfg_date, expiry_date, storage_temp_min, storage_temp_max}
→ pydantic validation + date normalization → compliance flags (expired, expiring <30 days, cold storage required, missing batch)

## Rules
- I know JS/Java but I'm new to Python. Explain new Python concepts briefly.
- Work phase by phase. STOP after each phase and tell me how to test it.
- Windows + PowerShell. Use a venv in `.venv`.
- Secrets live in `.env` and are never committed. Add `.env` and `.venv` to .gitignore.
- Keep code simple and readable. No over-engineering, the deadline is tight.
- Save preprocessing before/after images so the UI can show them.