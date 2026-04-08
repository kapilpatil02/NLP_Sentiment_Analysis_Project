import streamlit as st
import pickle
import requests
#from vadarsentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
print(analyzer.polarity_scores("This is amazing"))


vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("sentiment_model.pkl", "rb"))


analyzer = SentimentIntensityAnalyzer()


def extract_sentiment_words(text, analyzer):
    words = text.split()
    
    positive_words = []
    negative_words = []

    for word in words:
        score = analyzer.polarity_scores(word)['compound']
        
        if score > 0:
            positive_words.append(word)
        elif score < 0:
            negative_words.append(word)

    return positive_words, negative_words



st.title("Product Review Sentiment Analyzer")

st.write("Analyze reviews using ML + VADER +  API Prediction")

review = st.text_area("Enter your review")


if st.button("Analyze Review"):

    if review.strip() != "":

        # =====================
        # ML MODEL
        # =====================
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == 2:
            sentiment = "Positive"
        elif prediction == 1:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"

        st.subheader("ML Model Prediction")
        st.success(f"{sentiment}")

        # =====================
        # VADER ANALYSIS
        # =====================
        vader_scores = analyzer.polarity_scores(review)

        st.subheader("VADER Sentiment Analysis")
        st.write(vader_scores)

        compound = vader_scores['compound']

        if compound >= 0.05:
            vader_sentiment = "Positive"
        elif compound <= -0.05:
            vader_sentiment = "Negative"
        else:
            vader_sentiment = "Neutral"

        st.info(f"VADER Sentiment: {vader_sentiment}")

        # =====================
        # GRAPH
        # =====================
        st.subheader("Sentiment Distribution")

        labels = ['Positive', 'Neutral', 'Negative']
        values = [
            vader_scores['pos'],
            vader_scores['neu'],
            vader_scores['neg']
        ]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Score")

        st.pyplot(fig)

        # =====================
        # WORD ANALYSIS
        # =====================
        st.subheader("Word-Level Sentiment")

        positive_words, negative_words = extract_sentiment_words(review, analyzer)

        col1, col2 = st.columns(2)

        with col1:
            st.success("Positive Words")
            st.write(positive_words if positive_words else "None")

        with col2:
            st.error("Negative Words")
            st.write(negative_words if negative_words else "None")

        # =====================
        # API CALL
        # =====================
        st.subheader("API Prediction")

        try:
            url = "http://127.0.0.1:8000/predict"
            
            response = requests.post(url, json={"review": review})

            st.write("Status Code:", response.status_code)

            result = response.json()
            st.write("API Raw Response:", result)

            if "sentiment" in result:
                st.success(f"API Sentiment: {result['sentiment']}")
            else:
                st.error("API Error:")
                st.write(result)

        except Exception as e:
            st.error(f"API Error: {e}")

        # =====================
        # TEXT ANALYSIS
        # =====================
        st.subheader("Text Analysis")

        word_count = len(review.split())
        char_count = len(review)

        st.write(f"Words: {word_count}")
        st.write(f"Characters: {char_count}")

        # =====================
        # INSIGHTS (MOST IMPORTANT)
        # =====================
        st.subheader("Insights")

        if word_count < 5:
            st.warning("Short review → prediction may be less reliable")

        if vader_scores['pos'] > 0.2 and vader_scores['neg'] > 0.2:
            st.info("Mixed sentiment detected")

        if sentiment != vader_sentiment:
            st.error("Model and VADER disagree → possible ambiguity or model limitation")

        if abs(compound) >= 0.6:
            st.success("Strong sentiment detected")

        if abs(compound) < 0.2:
            st.info("Low confidence sentiment")

        if vader_sentiment == "Neutral":
            st.warning("Review is neutral → unclear opinion")

        if word_count > 100:
            st.info("Long review → may contain mixed opinions")

    else:
        st.warning("Please enter a review")