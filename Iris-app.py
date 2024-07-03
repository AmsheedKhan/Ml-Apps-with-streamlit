

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.write("""
 Simple Iris Flower Prediction App

 This app predicts the **Iris Flower** type!""")
st.sidebar.header('User Interface Parameters')

def user_input_features():
  sepal_length=st.sidebar.slider('Sepal Length',4.3,7.9,5.8)
  sepal_width=st.sidebar.slider('Sepal Width',2.0,4.4,3.8)
  petal_length=st.sidebar.slider('Petal_length',1.0,6.9,1.3)
  petal_width=st.sidebar.slider('Petal_length',0.1,2.5,0.2)
  data={'sepal_length':sepal_length ,
        'sepal_width':sepal_width ,
        'petal_lenth':petal_length,
        'petal_width':petal_width }
  features=pd.DataFrame(data,index=[0])
  return features

df=user_input_features()

st.subheader('User input parameters')
st.write(df)

iris= load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

rfc=RandomForestClassifier(n_estimators=100,criterion="gini")
rfc.fit(x_train,y_train)
rfc.score(x_test,y_test)

prediction=rfc.predict(df)
prediction_proba=rfc.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)
st.subheader('The predictions')
st.write(iris.target_names[prediction])
st.subheader('The probablity')
st.write(prediction_proba)

