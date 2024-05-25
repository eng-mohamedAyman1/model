from flask import Flask, jsonify
import pickle
import re
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')  # Ensure NLTK resources are available

app = Flask(__name__)

# Load the model and TF-IDF vectorizer
try:
    model = pickle.load(open('saved_model.pkl', 'rb'))
    cv = pickle.load(open('saved_tfidf.pkl', 'rb'))
except Exception as e:
    model = None
    cv = None
    print(f"Error loading model or vectorizer: {e}")

# Main route
@app.route('/')
def hello_world():
    return "Hello, world!"

# Prediction route
@app.route("/<data>")
def predict(data):
    # Handle if data is null
    if not data:
        return jsonify({"error": "Missing review text in request body"}), 400

    if not model or not cv:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500

    review_text = data

    # Process the review (cleaning, TF-IDF conversion)
    review = re.sub('[^a-zA-Z]', ' ', review_text)  # Remove unnecessary items
    review = review.lower()  # Lowercase
    ps = PorterStemmer()
    review = ' '.join([ps.stem(word) for word in review.split()])
    review_vector = cv.transform([review]).toarray()

    # Predict sentiment
    prediction = model.predict(review_vector)
    sentiment = "Negative" if prediction == 0 else "Positive"

    # Return the predicted sentiment
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
