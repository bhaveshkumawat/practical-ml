import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('./pcamodel.pkl', 'rb')) 
#model_randomforest = pickle.load(open('/content/drive/My Drive/machine learning/lab5/randomforest.pkl', 'rb')) 
#dataset= pd.read_csv('/content/drive/My Drive/machine learning/midterm2/PCA and NN Dataset2.csv')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
def predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange):
  output= model.predict(sc.transform([[meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange]]))
  if output==[1]:
    prediction="Person is male"
  else:
    prediction="person is female"
  print(prediction)
  return prediction

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab PCA Experiment: PIET18CS035</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Gender Prediction using PCA Algorithm")

    meanfreq = st.number_input('Insert meanfreq')

    sd = st.number_input('Insert sd')

    median =  st.number_input('Insert median')

    iqr  = st.number_input('Insert iqr')

    skew   = st.number_input('Insert skew')

    kurt  = st.number_input('Insert kurt')
    
    mode = st.number_input('Insert mode')
    centroid = st.number_input('Insert centroid')
    dfrange = st.number_input('Insert dfrange')
    #Age = st.text_input("Age","Type Here")
    
    resul=""
    if st.button("PCA Prediction"):
      result=predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange)
      st.success('PCA Model has predicted {}'.format(result))
    
    if st.button("About"):
      st.header("Developed by Bhavesh Kumawat")
      st.subheader("Student , Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Mid term 2 Exam</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
