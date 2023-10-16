# Loan-Approval-Machine-Learning-Project

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



