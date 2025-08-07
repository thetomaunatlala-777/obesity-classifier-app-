import streamlit as st
import pandas as pd


st.write(""" # Obesity Classifier App
         """)



st.markdown("""
    <p>
    Shown is the complete Data Science pipeline, done by the AnalytiQ team from the University of the Witwatersrand!
    </p>
    """, unsafe_allow_html=True)


st.markdown("""
    <h3>Problem Statement</h3>
    <p>
    Obesity is a chronic complex disease that is affecting many individuals worldwide. It is defined by excessive fat deposits that can impair health. Obesity can lead to increased risk of type 2 diabetes and heart disease, it can affect bone health and reproduction, it increases the risk of certain cancers. Obesity influences the quality of living, such as sleeping or moving. 
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
            <p>
            The goal is to use health metrics such as Age, Body Mass Index (BMI), Gender, Weight and Height to classify an individual's weight status (into categories namely: obese, overweight, underweight or normalweight.
            </p>
            """, unsafe_allow_html=True)


##Loading the dataset

df=pd.read_csv("Obesity Classification 2.csv")

