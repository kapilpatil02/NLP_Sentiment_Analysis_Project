from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model + vectorizer
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("sentiment_model.pkl", "rb"))

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#vectorizer = pickle.load(open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb"))
#model = pickle.load(open(os.path.join(BASE_DIR, "sentiment_model.pkl"), "rb"))

# Input schema
class ReviewInput(BaseModel):
    review: str

# Label mapping
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: ReviewInput):
    try:
        review = data.review

        # Transform text
        vec = vectorizer.transform([review])

        # Predict
        pred = model.predict(vec)[0]

        # Convert to label
        sentiment = label_map.get(pred, str(pred))

        return {
            "review": review,
            "sentiment": sentiment
        }

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}