import numpy as np
import pickle
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
st.title("Twitter Cyberbulling Detection System")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1613062975324-f95b97f29233?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
def predict(text):

    loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    x = loaded_vectorizer.transform([text]).toarray() # converting text to bag of words model (Vector)
    pred = loaded_model.predict(x) # predicting the language
    if pred == 0:
        pred = "has cyberbullying tone"
    
    elif pred == 1:
        pred = "is Neutral"
    st.write("This tweet: ", pred) # printing th
     
a= st.text_input("Enter your Sentence: ")
if st.button("Predict"):
    predict(a)

def main():

    st.title("Report this tweet")
    st.text_input("Enter your Report: ")

    if st.button("Submit Report"):
        st.success("Your Report has been successfully submitted to twitter \n\n Thank you for your feedback!")

if __name__ == '__main__':
    main()
