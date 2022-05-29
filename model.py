import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle

df=pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')
label_encoder = preprocessing.LabelEncoder()
df['Sex']= label_encoder.fit_transform(df['Sex'])
df['Age'] = df['Age'].fillna(0)
x= df.drop(['PassengerId', 'Survived','Name','Cabin', 'Ticket','Embarked'],axis=1)
y=df['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=3)
dt2=DecisionTreeClassifier()

scalar = StandardScaler()
x_transform = scalar.fit_transform(x)
dt2=DecisionTreeClassifier(criterion='gini',max_depth=20,min_samples_leaf=4, min_samples_split=8,splitter='random')
dt2.fit(x_train,y_train)

pickle.dump(dt2,open('final_model.pkl','wb'))
loadmodel=pickle.load(open('final_model.pkl','rb'))