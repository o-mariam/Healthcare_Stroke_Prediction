import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model


data=pd.read_csv('healthcare-dataset-stroke-data.csv', dtype={col: np.float32 for col in [ 'bmi']})
#data=data.drop(['smoking_status'],axis=1)
#print(data)

data=pd.get_dummies(data)

data_log=data.copy(deep=True)
print(data_log)
rows=data_log.loc[data_log['bmi'].isnull()]

data_log=data_log.drop(rows.index)
#print(data_log)
X_log=data_log.drop('bmi',axis=1)
Y_log=data_log['bmi']
#print(Y_log)


X=data.copy(deep=True)
X=data.drop(['stroke'],axis=1) #drop the stroke column 
Y=data['stroke']

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X_log, Y_log, test_size = 0.25) ##split data 

index_list =rows.index

regr = linear_model.LinearRegression().fit(X_train, Y_train)


rows=rows.drop('bmi',axis=1)
bmi_pred = regr.predict(rows)

bmi_pred=pd.DataFrame(({'bmi': bmi_pred[:]}))

bmi_pred=bmi_pred.set_index([pd.Index(index_list)])


rows['bmi']=bmi_pred['bmi']

data_log= data_log.append(rows, ignore_index=True)
#print(data_log)

#print(data_log.isnull())

X=data_log.copy(deep=True)
X=data_log.drop(['stroke'],axis=1) #drop the stroke column 
Y=data_log['stroke']


from imblearn.over_sampling import SMOTE
smote=SMOTE()
x_smote,y_smote=smote.fit_resample(X,Y)

X_train,X_test, Y_train, Y_test = train_test_split(x_smote,y_smote, test_size = 0.25) ##split data 



from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,class_weight='balanced')
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("\n\n\n\n\n************Confusion Matrix************\n")
print(confusion_matrix(Y_test,Y_pred))


target_names = ['0', '1'] # Optional display names matching the labels
print("\n\n\n\n\n************Report************\n")
print(classification_report(Y_test,Y_pred, target_names=target_names)) # Text summary of the precision, recall, F1 score for each class
