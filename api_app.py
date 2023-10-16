from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import json
import pandas as pd


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
    
model = pickle.load(open('model/model.sav','rb'))

@app.get('/')
async def index():
    return {'message':"hii, welcome to loan approval prediction api"}

@app.post('/predict')
async def predict_loan(inputs : model_input):
    input_data = inputs.model_dump_json()
    input_dict = json.loads(input_data)
    no_of_dependents = input_dict["no_of_dependents"]
    education = input_dict["education"]
    self_employed = input_dict["self_employed"]
    income_annum = input_dict["income_annum"]
    loan_amount = input_dict["loan_amount"]
    loan_term = input_dict["loan_term"]
    cibil_score = input_dict["cibil_score"]
    residential_assets_value = input_dict["residential_assets_value"]
    commercial_assets_value = input_dict["commercial_assets_value"]
    luxury_assets_value = input_dict["luxury_assets_value"]
    bank_asset_value = input_dict["bank_asset_value"]
    
    preds = model.predict([[no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value]])
    return {'prediction':preds}