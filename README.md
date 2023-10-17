# Loan-Approval-Machine-Learning-Project

![banner](https://github.com/Hg03/Loan-Approval-Machine-Learning-Project/blob/main/assets/20231017_202259_0000.png)

## Exploratory Data Analysis using sweetviz library

```python
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sweetviz
from sklearn import model_selection

data = pd.read_csv('https://raw.githubusercontent.com/Hg03/Loan-Approval-Machine-Learning-Project/main/data/loan_data.csv')
train, test = model_selection.train_test_split(data.drop(['loan_status'],axis=1), test_size=0.3,shuffle=42,stratify=data.loan_status_encoded)
report = sweetviz.analyze([train,"Train"],target_feat='loan_status_encoded')
report.show_html("Report.html")
```

### Output

https://github.com/Hg03/Loan-Approval-Machine-Learning-Project/assets/69637720/e8f73208-ca35-4c57-b606-c631e1e8a50d

## Build Preprocessing pipeline with integration of Random Forest model

```python
from sklearn import impute, preprocessing, metrics, compose, pipeline, tree, ensemble, linear_model
from category_encoders import BinaryEncoder

x_train,x_test,y_train,y_test = train.drop(['loan_status_encoded'],axis=1),test.drop(['loan_status_encoded'],axis=1),train.loan_status_encoded,test.loan_status_encoded
numerical_features = x_train.select_dtypes(include='number').columns
categorical_features = x_train.select_dtypes(include='object').columns
ordinal_features = categorical_features
## Build the preprocessing pipeline
num_imputer = impute.SimpleImputer(strategy = 'mean')
cat_imputer = impute.SimpleImputer(strategy = 'most_frequent')
cat_encoder = BinaryEncoder()
p1 = compose.make_column_transformer((num_imputer,numerical_features),(cat_imputer,['education','self_employed']),remainder='passthrough')
p2 = compose.make_column_transformer((cat_encoder,[-1,-2]),remainder='passthrough')
preprocessing_pipeline = pipeline.make_pipeline(p1,p2)

## integrate random forest model with pipeline
rf = ensemble.RandomForestClassifier()
rf_pipeline = pipeline.make_pipeline(preprocessing_pipeline,rf)
rf_pipeline.fit(x_train,y_train)
```

### Score we got
|Model|Score|
|-----|-----|
|Random Forest Classifier|0.9617329345097158|

## Save the model and use it

```python
import joblib
joblib.dump(rf_pipeline,'model.sav')
sample = data.head(1) # sample to test
model.predict(sample) # array([1])
```

## Build API using FastAPI

### We'll build the schema for input that we are going to pass through the model

```python
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
```

### Load the pickles or saved model

```python
with open('model/model.pkl','rb') as f:    
    model = pickle.load(f)
```

### Create an endpoint for entry and prediction

```python
@app.get('/')
async def index():
    return {'message':"hii, welcome to loan approval prediction api"}

@app.post('/predict')
async def predict_loan(inputs : model_input):
    input_data = inputs.model_dump_json()
    to_feed = pd.DataFrame([inputs.model_dump().values()],columns=inputs.model_dump().keys())
    preds = model.predict(to_feed)
    return {'prediction':int(preds)}
```

### Run the command

`uvicorn api_app:app` to run the api and go to `localhost:8000/docs`



