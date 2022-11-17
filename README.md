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
Developed by  : SANJAY S
RegisterNumber: 22007761 
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

![image](https://user-images.githubusercontent.com/115128955/202491937-624f3b93-d434-4411-b6b1-2e37ab23bc4e.png)

![image](https://user-images.githubusercontent.com/115128955/202491992-f7176afe-774f-428f-ad1c-074fcbdec43c.png)

![image](https://user-images.githubusercontent.com/115128955/202492034-0b669bc2-ded6-4fbd-bb30-00e99de5cbe4.png)

![image](https://user-images.githubusercontent.com/115128955/202492107-58acf22d-c572-498e-9841-fc420a70908a.png)

![image](https://user-images.githubusercontent.com/115128955/202492157-b5677d73-2b78-4392-b068-c79e779e0343.png)

![image](https://user-images.githubusercontent.com/115128955/202492199-73c3b47c-cb36-42f8-a643-0dec94a3442a.png)

![image](https://user-images.githubusercontent.com/115128955/202492248-7a21518c-b73b-467c-be96-1c9a5b468ec2.png)

![image](https://user-images.githubusercontent.com/115128955/202492284-e68c782b-613e-47fd-b1ca-25d8c7bb5ffd.png)

![image](https://user-images.githubusercontent.com/115128955/202492322-04df4e99-bb15-4975-9e13-7f53993e9f38.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
