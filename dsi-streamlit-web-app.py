
# import libraries

import streamlit as st
import pandas as pd
import joblib # to save our model pipeline

# load the ML model pipeline object
model = joblib.load('model.joblib')

# add title and instruction

st.title('Purchase Prediction ML Model')
st.subheader('Enter customer info & submit for likelihood to purchase')

# add age input form
# use Streamlit Documentation> API reference> Input widgets
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)  # this is the default value displayed, otherwise it will show 0



# add gender input form
gender = st.radio(
    label = "02. Enter the customer's gender", 
    options = ['M', 'F'])
    

# add credit score input form
credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500)  # this is the default value displayed, otherwise it will show 0

# submit inputs to model
if st.button('submit for prediction'):
    # store our data in a dataframe for prediction
    df = pd.DataFrame({"age":[age], "gender":[gender], "credit_score": [credit_score]})
    
    # apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(df)[0][1]
    
    # output the prediction
    st.subheader(f"Based on these customer attributes, this ML model predicts a purchase probability of {pred_proba:.0%}")
    # NOTE: adding :.0% means that it returns a whole percentage number instead of a decimal
    