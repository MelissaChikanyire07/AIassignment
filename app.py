# loading in the model to predict on the data
import pandas as pd
import numpy as np
import pickle
import streamlit as st
pickle_in = open('classifier.pkl', 'wb')
#classifier = pickle.load(pickle_in)


def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(seniorcitizen, partner,tenure,phoneservice,contract,paperlessbilling,monhtlycharges,totalcharges):
    prediction = classifier.predict(
        [[seniorcitizen, partners,tenure,phoneservice,contract,paperlessbilling,monhtlycharges,totalcharges]])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Customer Churn Prediction")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Churn Classifier ML App </h1>
    </div>
    """
    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    seniorcitizen= st.text_input("SeniorCitizen", " ")
    partner= st.text_input("Partner", " ")

    tenure = st.text_input("Tenure", " ")
    phoneservice  = st.text_input("PhoneService", " ")
    contract = st.text_input("Contract", " ")
    paperlessbilling = st.text_input("Paperlessbilling", " ")
    monthlycharges = st.text_input("MonthlyCharges", " ")
    totalcharges = st.text_input("TotalCharges", " ")
    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(seniorcitizen, partner, tenure,phoneservice,contract,paperlessbilling,monthlycharges,totalcharges)
    st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()