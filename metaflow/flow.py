from metaflow import FlowSpec, step, card
import pandas as pd
from sklearn import model_selection, impute, preprocessing, compose, pipeline

class Flow(FlowSpec):
    
    
    vals = [34.56,67]
    @card
    @step
    def start(self):
        self.next(self.load_data)
      
    @card  
    @step
    def load_data(self):
        self.data = pd.read_csv('data/loan_data.csv')
        self.features = self.data.drop(['loan_id','loan_status_encoded','loan_status'],axis=1)
        self.target = self.data.loan_status_encoded
        self.numerical_columns = [col for col in self.data.columns if self.data[col].dtype != 'O']
        self.categorical_columns = [col for col in self.data.columns if col not in self.numerical_columns]
        print('Data loaded successfully !!')
        self.next(self.split_up)
      
    @card    
    @step
    def split_up(self):
        self.x_train,self.x_test,self.y_train,self.y_test = model_selection.train_test_split(self.features,self.target, test_size=0.2,stratify = self.target)
        print('Data splitted up successfully !!')
        self.next(self.preprocess)
      
    @card    
    @step
    def preprocess(self):
        
        self.imputer = compose.make_column_transformer((impute.SimpleImputer(strategy='mean'),[0,3,4,5,6,7,8,9,10]), (impute.SimpleImputer(strategy='most_frequent'),[1,2]),remainder='passthrough')
        self.encoder = compose.make_column_transformer((preprocessing.OrdinalEncoder(),[-1,-2]),remainder='passthrough')
        self.prep_pipeline = pipeline.make_pipeline(self.imputer,self.encoder)
        
        #print(self.prep_pipeline.fit_transform(self.x_train)[0])
        
        print('Pipeline for preprocessing is created !!')
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow is done!")

if __name__ == "__main__":
    Flow()
