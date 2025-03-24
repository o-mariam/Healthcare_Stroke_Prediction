import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('healthcare-dataset-stroke-data.csv')
data=data.drop(['smoking_status'],axis=1)


data=pd.get_dummies(data)

median=data['bmi'].median()
data['bmi'].fillna(median,inplace=True)

X=data.drop(['stroke'],axis=1) #drop the sroke column
Y=data['stroke']

from imblearn.over_sampling import SMOTE
smote=SMOTE()
x_smote,y_smote=smote.fit_resample(X,Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_smote,y_smote, test_size = 0.25) ##split data 

 

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("\n\n\n\n\n************Confusion Matrix************\n")
print(confusion_matrix(Y_test,Y_pred))


target_names =  ['0', '1'] # Optional display names matching the labels
print("\n\n\n\n\n************Report************\n")
print(classification_report(Y_test,Y_pred, target_names=target_names)) # Text summary of the precision, recall, F1 score for each class
