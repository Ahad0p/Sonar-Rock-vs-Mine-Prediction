from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("artifact/model.pkl")
preprocessor = joblib.load("artifact/preprocessor.pkl")
label_encoder = joblib.load("artifact/label_encoder.pkl")

class InputData(BaseModel):
    data: list

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.data)
    transformed = preprocessor.transform(arr)
    pred = model.predict(transformed)
    label = label_encoder.inverse_transform(pred.astype(int))[0]
    return {"prediction": label}