from fastapi import FastAPI

app = FastAPI(title="ColdLens API")


@app.get("/health")
def health():
    return {"status": "ok"}
