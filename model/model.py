
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle=True)

from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import RandomForestClassifier

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

cls = RandomForestClassifier(criterion='entropy', n_estimators=300, random_state=42)
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)

print('Accuracy is',cls.score(X_test,y_test)*100,'%')

filename = 'finalised_model.sav'
joblib.dump(cls, filename)


