import os
from dotenv import load_dotenv
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('loan_data.csv')

""" Configuring Appwrite Client For API Access """
# Instantiating Appwrite Client
client = Client()

# To load environment variables
load_dotenv()

(client
 # Setting API Endpoint
 .set_endpoint('https://cloud.appwrite.io/v1')
 # Setting Project ID
 .set_project(os.getenv("PROJECT_ID"))
 # Setting API Key
 .set_key(os.getenv("API_KEY"))
 )

databases = Databases(client)
""" Configuration Code Ends Here """

"""Creating Database"""
# To generate unique database ID
db_id = ID.unique()

create_db = databases.create(db_id, 'LoanDB')
print("Database Successfully Created.")
""" Database Creation Code Ends Here """

"""Creating Collections"""
# Database ID
database_id = create_db['$id']
# For Generating Unique Collection ID
collection_id = ID.unique()

# Creating a New Collection
new_collection = databases.create_collection(database_id=database_id,
                                             collection_id=collection_id,
                                             name='Loans')

print('Collection Successfully Created.')
""" Collection Creation Code Ends Here """

"""Creating Attributes"""
# Collection ID of Book
c_id = new_collection['$id']

no_of_dependents = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key="no_of_dependents",required=True)

education = databases.create_string_attribute(database_id=database_id,collection_id=c_id,key="education",size=100,required=True)

self_employed = databases.create_string_attribute(database_id=database_id,collection_id=c_id,key="self_employed",size=100,required=True)

income_annum = databases.create_float_attribute(database_id=database_id,collection_id=c_id,key='income_annum',required=True)

loan_amount = databases.create_float_attribute(database_id=database_id,collection_id=c_id,key='loan_amount',required=True)

loan_term = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='loan_term',required=True)

cibil_score = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='cibil_score',required=True)

residential_assets_value = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='residential_assets_value',required=True)

commercial_assets_value = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='commercial_assets_value',required=True)

luxury_assets_value = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='luxury_assets_value',required=True)

bank_asset_value = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='bank_asset_value',required=True)

loan_status = databases.create_integer_attribute(database_id=database_id,collection_id=c_id,key='loan_status',required=True)



print("Attributes Successfully Created.")
""" Attribute Creation Code Ends Here """

""" Adding Documents """
# Unique Identifier for Document ID
document_id = ID.unique()

""" Function for Adding Documents(data) in the Database """


def add_doc(data):
    try:
        
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Adding Data"):
            data_dict = {"no_of_dependents": None,"education": None,"self_employed": None,"income_annum": None,"loan_amount": None,"loan_term": None,"cibil_score": None,"residential_assets_value": None,"commercial_assets_value": None,"luxury_assets_value": None,"bank_asset_value": None,"loan_status": None}
            data_dict['no_of_dependents'] = row['no_of_dependents']
            data_dict['education'] = row['education']
            data_dict['self_employed'] = row['self_employed']
            data_dict['income_annum'] = row['income_annum']
            data_dict['loan_amount'] = row['loan_amount']
            data_dict['loan_term'] = row['loan_term']
            data_dict['cibil_score'] = row['cibil_score']
            data_dict['residential_assets_value'] = row['residential_assets_value']
            data_dict['commercial_assets_value'] = row['commercial_assets_value']
            data_dict['luxury_assets_value'] = row['luxury_assets_value']
            data_dict['bank_asset_value'] = row['bank_asset_value']
            data_dict['loan_status'] = row['loan_status_encoded']
            document = data_dict
            doc = databases.create_document(database_id=database_id,collection_id=c_id,document_id=document_id,data=document)

        

    except Exception as e:
        print(e)
        print("Something went wrong, exiting the program.")




# Calling function with the data to be added
add_doc(data)
print("Documents Successfully Added.")
