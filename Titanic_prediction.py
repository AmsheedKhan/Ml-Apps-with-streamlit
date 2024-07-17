
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("""# Titanic prediction :ship: """)
titanic=sns.load_dataset('titanic')
gender = st.radio("Enter the gender",['Male :boy:','Female :girl:'])
gender_map={'Male :boy:':1,'Female :girl:':1}[gender]
class_num=st.radio("Enter the class of the passenger ",['1','2','3'])
age=st.slider("Enter the age of the passenger",1,100)
town=st.radio("Enter the Town of the passenger",['Cherbourg','Queenstown','Southampton'])
if town=='Cherbourg':

    embark_town_Cherbourg=1
else:
    embark_town_Cherbourg=0
if town=='Queenstown':
    embark_town_Queenstown=1
    
else:
    embark_town_Queenstown=0
from word2number import w2n
class_map={'First':1,'Second':2,'Third':3}
titanic['class_num']=titanic['class'].map(class_map)

dummy=pd.get_dummies(titanic[['sex','embark_town']])
dummy=dummy.astype(int)

titanic=pd.concat([titanic,dummy],axis=1)
titanic=titanic.drop(['sex','embark_town','class'],axis=1)
titanic=titanic.drop(['sex_male','embark_town_Southampton'],axis=1)

titanic['age']=titanic['age'].fillna(titanic['age'].mean())
x1=titanic[['sex_female','class_num','age','embark_town_Cherbourg','embark_town_Queenstown']]
y=titanic['survived']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2)
r=RandomForestClassifier(n_estimators=10)
r.fit(x_train,y_train)
r.score(x_test,y_test)
g=[[gender_map,class_num,age,embark_town_Cherbourg,embark_town_Queenstown]]
r1=r.predict(g)
rp=r.predict_proba(g)
if r1==1:
    st.title('Survived :relieved: :+1:')
elif r1==0:
    st.title('Not survived 	:disappointed: :thumbsdown:')
else:
    st.write('Not enough data')
        
st.write("The probablity would be",rp)