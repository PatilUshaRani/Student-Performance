import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Student Performance Prediction App")

# Section 1: Load Data
st.header("1. Load Dataset")
df = pd.read_csv('Student_Performance.csv')
st.write("Dataset Preview:")
st.dataframe(df.head())

# Section 2: Dataset Overview
st.header("2. Dataset Overview")
st.subheader("Tail of Dataset")
st.dataframe(df.tail())

st.subheader("Descriptive Statistics")
st.write(df.describe())

st.subheader("Info")
buffer = []
df.info(buf=buffer)
s = '\n'.join(buffer)
st.text(s)

# Section 3: Preprocessing
st.header("3. Data Preprocessing")
st.write("Encoding 'Extracurricular Activities' column...")
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
st.write(df.head())

# Section 4: Model Training
st.header("4. Train-Test Split")
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.header("5. Linear Regression Model")
regression = LinearRegression()
regression.fit(X_train, y_train)
y_predict = regression.predict(X_test)

# Section 5: Results
st.header("6. Model Evaluation")
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

st.write("### R² Score:", round(r2 * 100, 2))
st.write("### RMSE:", round(rmse, 2))

st.subheader("Predicted Values")
st.write(y_predict)




