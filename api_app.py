from fastapi import FastAPI
from pydantic import BaseModel
import pickle5 as pickle
import uvicorn
import json
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


app = FastAPI()

class model_input(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int

with open('model/model.pkl','rb') as f:    
    model = pickle.load(f)

@app.get('/')
async def index():
    return {'message':"hii, welcome to loan approval prediction api"}

@app.post('/predict')
async def predict_loan(inputs : model_input):
    input_data = inputs.model_dump_json()
    to_feed = pd.DataFrame([inputs.model_dump().values()],columns=inputs.model_dump().keys())
    preds = model.predict(to_feed)
    return {'prediction':int(preds)}