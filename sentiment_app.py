import streamlit as st
from transformers import pipeline

# Set the title of the app
st.title("Sentiment Analysis App")
st.write("Enter some text and the app will analyze its sentiment!")

# Load the pre-trained sentiment analysis model
@st.cache_resource  # Cache the model to avoid reloading it every time
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# Create a text input box
user_input = st.text_area("Enter your text here:", "I love using Streamlit for building apps!")

# Analyze the sentiment when the user clicks the button
if st.button("Analyze Sentiment"):
    if user_input:
        # Perform sentiment analysis
        result = model(user_input)[0]
        sentiment = result['label']
        confidence = result['score']

        # Display the result
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence: **{confidence:.2f}**")
    else:
        st.write("Please enter some text to analyze.")