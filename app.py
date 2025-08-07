import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv('Student_Performance.csv')
df
df.head()
df.tail()
df.describe()
df.info()
lb = LabelEncoder()
lb.fit_transform(df['Extracurricular Activities'])
df
X = df.drop('Performance Index', axis= 1)
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
y_predict = regression.predict(X_test)
y_predict
print(y_predict)
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print('r2 Score = ', r2*100)
print('RMSE=', rmse)


