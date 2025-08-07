import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


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


st.write(""" ### 3. Visualisations
         """)


num_cols = ['Age', 'Height', 'Weight', 'BMI']

fig, axes = plt.subplots(len(num_cols) // 2, 2, figsize=(12, 8))  
axes = axes.flatten()


for i in range(0, len(num_cols), 2):
    col1, col2 = st.columns(2)  
    
   
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df[num_cols[i]], bins=20, color='lightblue', edgecolor='black')
        ax.set_title(f"{num_cols[i]} Distribution")
        ax.set_xlabel(num_cols[i])
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    if i + 1 < len(num_cols):
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df[num_cols[i + 1]], bins=20, color='lightblue', edgecolor='black')
            ax.set_title(f"{num_cols[i + 1]} Distribution")
            ax.set_xlabel(num_cols[i + 1])
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

st.markdown(""" 
- The histograms of Age, Height, Weight, and BMI help identify skewed data, outliers, and imbalanced distributions in the dataset.  
- The graphs provide an immediate visual comparison, showing which bars are close or far apart in height. This observable pattern can be useful for identifying similarities between features.  
""")


fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(data=df, x="Label", y="Age", palette="crest")

ax.set_title("Age Distribution Across Obesity Categories")
ax.set_xlabel("Obesity Category (Label)")
ax.set_ylabel("Age")
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
- This boxplot compares the ages within each obesity category, showing differences in median age and age range.
- We can see that the median age for obese individuals is between 50-60 years of age, whilst underweight individuals is between 25-35 years of age
""")



num_cols = ['Weight', 'BMI'] 


selected_col = st.selectbox("Select attribute to compare against Obesity Category (Label)", num_cols)


agg_df = df.groupby("Label")[selected_col].mean().reset_index()

fig = px.bar(
    agg_df, 
    x="Label", 
    color_discrete_sequence=['lightblue'],
    y=selected_col, 
    title=f"{selected_col} by Obesity Levels"
)

st.plotly_chart(fig, use_container_width=True)
