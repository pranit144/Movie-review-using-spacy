import streamlit as st
import joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Load the saved model
model = joblib.load('sentiment_model.sav')

# Load SpaCy's small English model for text processing
nlp = spacy.load("en_core_web_sm")

# Gather stopwords
stopwords = list(STOP_WORDS)
punct = string.punctuation


def dataCleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    clean_tokens = [token for token in tokens if token not in punct and token not in stopwords]
    return clean_tokens


# Streamlit UI
st.title("Sentiment Analysis with SpaCy")
st.write("Enter a review text to predict its sentiment")

# Input text box for user input
user_input = st.text_area("Enter Review:")

if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_text = " ".join(dataCleaning(user_input))

        # Make prediction using the loaded model
        prediction = model.predict([cleaned_text])

        # Display the result
        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
    else:
        st.warning("Please enter a review text to predict.")
