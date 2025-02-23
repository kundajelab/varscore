from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import pandas as pd
from varscore.prioritization import prioritize_variants
from varscore.annotations import (
    VariantAnnotationInput,
    AnnotatedVariant,
    annotate_variant,
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:8000",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    A simple root endpoint to check if the server is running.
    """
    return {"message": "Hello World"}


@app.post("/annotate")
async def annotate(input: VariantAnnotationInput) -> AnnotatedVariant:
    return annotate_variant(input)


@app.post("/prioritize/")
async def predict(input_data: List[dict]) -> List[dict]:
    df = pd.DataFrame(input_data)
    return prioritize_variants(df).to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
