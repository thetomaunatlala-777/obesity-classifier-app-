import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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


st.write(""" ### 1. Loading the data
         """)

st.markdown("""
            <p>
            The goal is to use health metrics such as Age, Body Mass Index (BMI), Gender, Weight and Height to classify an individual's weight status (into categories namely: obese, overweight, underweight or normalweight.
            </p>
            """, unsafe_allow_html=True)




##Loading the dataset

df=pd.read_csv("Obesity Classification 2.csv")


st.markdown("""
            ### Data Preview:
            """)

st.dataframe(df.head())

st.markdown("""From the above few samples of the dataset, we can see that there is no need for the ID column as it replicates the dataframe index column and should be removed due to irrelevance to modelling the data.
            """)

st.markdown("""
            ### Data Shape:
            """)

##Data shape
st.write("""There are 108 entries made against 7 attributes that determine an individuals obesity health profile.
         """)

st.markdown("""
            ### Data Info:
            """)

##Data Info
if st.checkbox("Click here to view the data info"):
    data_info=pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values
    })
    st.dataframe(data_info)

st.write("""The dataset includes a mixture of categorical and numerical attributes about an individual's health profile to determine their status of being healthy (normal weight) or obese (overweight, underweight, etc)
There are no missing/null values.
         """)


st.write(""" ### 2. Exploratory Data Analysis (EDA)
         """)

st.markdown("""
            ### Data Statistics:
            """)

st.dataframe(df.describe())


st.markdown("""
**From the above information, the following can be deduced:**

- The average individual is approximately **47 years old**, **166.5 cm** tall, weighs **59.5 kg**, and has a **BMI of 20.5 kg/m²**.
- The youngest individual is **11 years old**, **120 cm** tall, weighs **10 kg**, and has a **BMI of 3.9 kg/m²**.
- This suggests the individual may be malnourished.  
-  This makes me wonder: Are individuals with a low weight and BMI at risk of having a low weight status? Are there conditions that promote an individual's weight status?
""")



st.markdown("""
            ### Column names:
            """)

st.dataframe(df.dtypes)


num_cols = ['Age', 'Height', 'Weight', 'BMI']

df[num_cols].hist(bins=20, figsize=(12, 8), color='lightblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
st.pyplot(plt.show())