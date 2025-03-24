
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('healthcare-dataset-stroke-data.csv')


print(data.info())
print(data.describe())


#Plot with age,bmi,heart disease and avg_glucose_level
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
data.plot(kind="hist", y="age", bins=70, color="b", ax=axes[0][0])
data.plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[0][1])
data.plot(kind="hist", y="heart_disease", bins=6, color="g", ax=axes[1][0])
data.plot(kind="hist", y="avg_glucose_level", bins=100, color="orange", ax=axes[1][1])


#Pie with stroke values
labels =data['stroke'].value_counts(sort = True).index
sizes = data['stroke'].value_counts(sort = True)

colors = ["green","red"]
explode = (0.05,0) 
 
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)


plt.show()


