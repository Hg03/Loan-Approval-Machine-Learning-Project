import streamlit as st
from typing import List
import pandas as pd
from sklearn import *
import pickle5 as pickle

def make_form(cols : List[str]):
    input = pd.DataFrame(columns = cols)
    with st.form("submission-form"):
        input.no_of_dependents = [st.number_input("Number of Dependents",min_value=0,max_value=6)]
        input.education = [st.selectbox("Education",options=['Graduate','Not Graduate'])]
        input.self_employed = [st.selectbox("Self Employed or not ?",options=["Yes","No"])]
        input.income_annum = [st.number_input("Income per annum")]
        input.loan_amount = [st.number_input("Loan Amount")]
        input.loan_term = [st.number_input("Loan Term",min_value=0,max_value=20)]
        input.cibil_score = [st.number_input("Cibil Score",min_value=0,max_value=800)]
        input.residential_assets_value = [st.number_input("Residential Assets Value")]
        input.commercial_assets_value = [st.number_input("Commercial Assets Value")]
        input.luxury_assets_value = [st.number_input("Luxury Assets Value")]
        input.bank_asset_value = [st.number_input("bank_asset_value")]
        submitted = st.form_submit_button("Approved or not ?")
        if submitted:
            model = pickle.load(open("model/model.pkl",'rb'))
            preds = model.predict(input)
            if preds[0] == 0:
                st.error("Sorry, your Loan is not approved according to the data you provided",icon="🚨")
            else:
                st.success("According to the data, your Loan is approved",icon="✅")

def main():
    st.title(":green[Loan] Approval Prediction :blue[Machine Learning Project]")
    columns = ["no_of_dependents","education","self_employed","income_annum","loan_amount","loan_term","cibil_score","residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"]
    make_form(columns)
if __name__ == "__main__":
    main()