%%writefile app.py
import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/machine learning/midterm/svmmodel.pkl', 'rb'))
#model_randomforest = pickle.load(open('/content/drive/My Drive/machine learning/lab5/randomforest.pkl', 'rb'))
dataset= pd.read_csv('/content/drive/My Drive/machine learning/midterm/Classification Dataset2.csv')
# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, :-1]
# Extracting dependent variable:
y = dataset.iloc[:, -1]
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X["Gender"] = labelencoder_X.fit_transform(X["Gender"])
X=X.values
y=y.values
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean', fill_value=None, verbose=0, copy=True  )
imputer
#Fitting imputer object to the independent variables x.
imputer= imputer.fit(X[:, 2:7])
#Replacing missing data with the calculated mean value
X[:, 2:7]= imputer.transform(X[:, 2:7])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Gender,Glucose,BP,Skin,Insulin,BMI,Pfunc,Age):
  if Gender=="Male": Gender=1
  else: Gender=0
  output= model.predict(sc.transform([[Gender,Glucose,BP,Skin,Insulin,BMI,Pfunc,Age]]))
  print("Purchased", output)
  if output==[1]:
    prediction="Person will have disease"
  else:
    prediction="person will not have disease"
  print(prediction)
  return prediction

def main():

    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction using SVM Algorithm")

    Gender = st.selectbox(
    "Gender",
    ("Male", "Female", "Others")
    )
    Glucose = st.number_input('Insert Glucose Level',0,300)

    BP = st.number_input('Insert BP Level',50,200)

    Skin =  st.number_input('Insert Skin Thickness',0,100)

    Insulin  = st.number_input('Insert Insulin Level',0,300)

    BMI   = st.number_input('Insert BMI',0,100)

    Pfunc  = st.number_input('Insert Pedigree function',0,10)

    Age = st.number_input('Insert a Age',0,100)
    #Age = st.text_input("Age","Type Here")

    resul=""
    if st.button("SVM Prediction"):
      result=predict_note_authentication(Gender,Glucose,BP,Skin,Insulin,BMI,Pfunc,Age)
      st.success('SVM Model has predicted {}'.format(result))

    if st.button("About"):
      st.header("Developed by Bhavesh Kumawat")
      st.subheader("Student , Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment 5: Support Vector Machine and Random Forest</p></center>
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
