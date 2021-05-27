import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

df= pd.read_csv('kidney_disease.csv')

columns_to_retain=['sg','al','sc','hemo','pcv','wbcc','rbcc','htn','classification']
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
df = df.dropna(axis=0)

lb= LabelEncoder()
for col in df.columns:
    if(df[col].dtype==np.number):
        continue
    else:
        df[col]= lb.fit_transform(df[col])

X = df.drop(['classification'],axis=1)
y = df['classification']



x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle=True)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe=Pipeline([
    ('rescale',StandardScaler()),
    ('classifier',RandomForestClassifier())
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

accuracy = pipe.score(X_test, y_test)
print(accuracy)

print(pipe.predict([[1.01,0,0.6,11.4,20,0]]))

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(clf.predict([[1.01,0,0.6,11.4,20,0]]))

filename = 'finalised_model1.sav'
joblib.dump(clf, filename)