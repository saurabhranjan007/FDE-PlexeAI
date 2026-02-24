"""
FastAPI app: POST /predict with risk_probability, risk_level, top_features (SHAP).
"""

from fastapi import FastAPI

app = FastAPI(title="Low-Review Risk API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /predict, load model, SHAP explanations, sample curl in README
