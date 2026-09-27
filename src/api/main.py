from fastapi import FastAPI
from pydantic import BaseModel
from src.infer.predictor import predict_smart   

app = FastAPI(title="Retail Sentiment API")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    label = predict_smart(item.text)            
    return {"label": label}
