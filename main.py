from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from keras import models

app = FastAPI()

model = models.load_model('h5model.h5')

class InputData(BaseModel):
    input: list[float]

@app.post('/predict')
async def predict(data: InputData):
    input_data = np.array(data.input).reshape(1, -1)
    prediction = model.predict(input_data)
    return {'prediction': prediction.tolist()}
