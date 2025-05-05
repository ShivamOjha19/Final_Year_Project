from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.inference import load_model_and_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    equation: str  # 'burgers', 'conv_diff', 'convection'
    X: list  # list of [x, t] pairs

@app.post("/predict")
def predict(data: InferenceRequest):
    output = load_model_and_predict(data.equation, data.X)
    return {"output": output}
