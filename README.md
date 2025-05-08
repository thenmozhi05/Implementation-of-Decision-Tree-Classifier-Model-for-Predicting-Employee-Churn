# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HIRUTHIK SUDHAKAR
RegisterNumber:  2122232400054
*/
```
```python
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
### Data Head:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/d5cb75c8-1a61-40b1-9f55-9dd6e2f28fe9)

### Dataset info :
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/909d5c90-e8fc-4e80-a696-180be9736872)

### Null Dataset:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/955a342a-5f5c-4b9e-b092-08c418ca2f04)

### Values count in left column:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/1e5df3da-26d0-4310-b3c4-7ef739eb55c3)

### Dataset transformed head:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/c5309dd8-61d5-4bbc-9339-ce580509b787)

### x.head:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/f10c925a-3ac2-44ef-a6ec-1e15da68f1d8)
### Accuracy:

![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/723a87ad-58a9-4512-90d1-31cddd146a48)

### Data prediction:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972122/c51ce862-b453-4847-abcc-cfb56674ed3d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
