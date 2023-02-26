import streamlit as st
from matplotlib import image
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor(n_estimators=100, random_state=0)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
dir_of_interest = os.path.join(PARENT_DIR, "resources")
IMAGE_PATH = os.path.join(dir_of_interest, "images", "download.jpeg")
DATA_PATH = os.path.join(dir_of_interest, "data", "metro-areas.csv")

st.title(":blue[Metro Politian Areas House Price Prediction] :house:")

img = image.imread(IMAGE_PATH)
st.image(img)
df=pd.read_csv(DATA_PATH)
st.write("Below is the Dataset used for Prediction.")
st.dataframe(df)
city=st.selectbox("Select City:",sorted(df["City"].unique()))
s_city=df[df['City']==city]['Location']
loc=st.selectbox("Select Location:",sorted(s_city.unique()))
col1,col2=st.columns(2)
fig_1=px.histogram(df[df['City']==city], x="City")
col1.plotly_chart(fig_1, use_container_width=True)
fig_2=px.histogram(df[df['Location']==loc], x="Location")
col2.plotly_chart(fig_2, use_container_width=True)
st.bar_chart(df[df["City"]==city]["BHK"])
area=st.slider('Select an Area in the range',1000,4500)
bhk=st.selectbox('Select No Of BHK:',sorted(df["BHK"].unique()))
par=st.radio("Do You Want Parking Space:",('Yes', 'No'))
if par=='Yes':
    par=1
else:
    par=0
btn=st.button("Predict")
if btn==True:
    newdf=df[df["Location"]==loc]
    x=newdf.iloc[:,2:5]
    y=newdf.iloc[:,5]
    model.fit(x,y)
    predvalue=[area,bhk,par]
    k=model.predict([predvalue,])[0]
    st.balloons()
    st.write('House Would cost Around:  ₹',str(int(k)))
    url="https://www.google.com/maps/place/"
    sp="+".join(loc.split())
    map=url+sp
    
    st.write("To View Location Click 👉 [Here](%s)" % map)
        
else:
    st.write('Click Predict Button')




