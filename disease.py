
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data=pd.read_csv('healthcare-dataset-stroke-data.csv')
#print(data.head())
#print(data.describe())

#data["stroke"].plot(kind="hist")
#data.hist()
#plt.show()  
#X=data["age"]
#Y=data["stroke"]
#plt.bar(X,Y)
#plt.show()

#print(data['bmi'].isnull()) #bgazei me true or false an exv nan values 

#print(data['bmi'])

#missing_values=["n/a","na","--"] #unexpected nan values
#df=pd.read_csv('healthcare-dataset-stroke-data.csv',na_values=missing_values)
#print(data['bmi'].isnull())

#print(data.isnull().values.any())
#print(data.isnull().sum().sum())
print(data['stroke'].isnull())