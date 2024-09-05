import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Title of the app
st.title("Sentiment Analysis App")

# Provide the full file path directly
import streamlit as st
import pandas as pd
import os

# File path
file_path = 'generic_sentiment_dataset_50k.csv'

if os.path.exists(file_path):
    # Load your dataset for training
    df = pd.read_csv(file_path)
    st.write("### Preview of the Dataset")
    st.write(df.head())
else:
    st.error("Data file not found. Please upload the file or check the path.")
    st.stop()


try:
    # Load your dataset for training
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Data file not found. Please check the path and try again.")
    st.stop()

# Display the first few rows of the dataset
st.write("### Preview of the Dataset")
st.write(df.head())

# Check if 'text' and 'sentiment' columns exist
if 'text' not in df.columns or 'sentiment' not in df.columns:
    st.error("The dataset must contain 'text' and 'sentiment' columns.")
    st.stop()

# Handle missing values in 'text' column
df['text'] = df['text'].fillna('')

# Prepare the data
X = df['text']
y = df['sentiment']

# Convert text data into numerical data
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

# User input for sentiment analysis
st.write("### Write a Review")
user_input = st.text_area("Enter your review here:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Transform user input using the trained vectorizer
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_vectorized)

        # Display the result
        st.write("### Sentiment Analysis Result")
        if prediction[0] == 'positive':
            st.success("The sentiment is Positive! 😊")
        elif prediction[0] == 'neutral':
            st.info("The sentiment is Neutral. 😐")
        else:
            st.error("The sentiment is Negative! 😔")
    else:
        st.warning("Please enter a review to analyze.")
