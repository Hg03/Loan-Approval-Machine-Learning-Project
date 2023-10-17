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
    # to_feed = pd.DataFrame(columns=["no_of_dependents","education","self_employed","income_annum","loan_amount","loan_term","cibil_score","residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"])
    # input_dict = json.loads(input_data)
    # to_feed["no_of_dependents"] = [input_dict["no_of_dependents"]]
    # to_feed["education"] = [input_dict["education"]]
    # to_feed["self_employed"] = [input_dict["self_employed"]]
    # to_feed["income_annum"] = [input_dict["income_annum"]]
    # to_feed["loan_amount"] = [input_dict["loan_amount"]]
    # to_feed["loan_term"] = [input_dict["loan_term"]]
    # to_feed["cibil_score"] = [input_dict["cibil_score"]]
    # to_feed["residential_assets_value"] = [input_dict["residential_assets_value"]]
    # to_feed["commercial_assets_value"] = [input_dict["commercial_assets_value"]]
    # to_feed["luxury_assets_value"] = [input_dict["luxury_assets_value"]]
    # to_feed["bank_asset_value"] = [input_dict["bank_asset_value"]]
    to_feed = pd.DataFrame([inputs.model_dump().values()],columns=inputs.model_dump().keys())
    preds = model.predict(to_feed)
    #preds = model.predict(to_feed)
    #print(preds)
    return {'prediction':int(preds)}