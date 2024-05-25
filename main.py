#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
# Function to load the model and TF-IDF vectorizer
def load_model_and_tfidf(model_filename="saved_model.pkl", tfidf_filename="saved_tfidf.pkl"):
  with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
  with open(tfidf_filename, 'rb') as file:
    cv = pickle.load(file)
  print(f"Loaded model from: {model_filename}")
  print(f"Loaded TF-IDF vectorizer from: {tfidf_filename}")
  return loaded_model, cv

# Example usage: Load the model and predict sentiment for a new review
loaded_model, cv = load_model_and_tfidf()

# new_review = "This restaurant provide a very good service."
# new_review_vector = cv.transform([new_review]).toarray()
# prediction = loaded_model.predict(new_review_vector)
# if prediction == 1:
#   print("Predicted sentiment: Negative")
# else:
#   print("Predicted sentiment: Positive")

from tkinter import Tk, Label, Entry, Button
import re
from nltk.stem.porter import PorterStemmer
def predict_sentiment(review):
  # Process the review (cleaning, TF-IDF conversion)
  review = re.sub('[^a-zA-Z]', ' ', review)  # Remove unnecessary items
  review = review.lower()  # Lowercase
  ps = PorterStemmer()
  review = ' '.join([ps.stem(word) for word in review.split()])
  review_vector = cv.transform([review]).toarray()

  # Predict sentiment
  prediction = loaded_model.predict(review_vector)
  if prediction == 1:
    return"Predicted sentiment: Negative"
  else:
    return"Predicted sentiment: Positive"


def handle_click():
  review_text = review_entry.get()
  sentiment = predict_sentiment(review_text)
  result_label.config(text=f"Sentiment: {sentiment}")


# Create the main window
root = Tk()
root.title("Sentiment Analysis")

# Set window size (replace with desired width and height)
root.geometry("500x300")  # Example: width 500 pixels, height 300 pixels

# Create labels
instruction_label = Label(root, text="Enter your review:")
instruction_label.pack()

# Create entry field for review
review_entry = Entry(root)
review_entry.pack()

# Create a button
predict_button = Button(root, text="Predict Sentiment", command=handle_click)
predict_button.pack()

# Create result label and position it below the entry field
result_label = Label(root, text="")
result_label.pack(pady=10)  # Add padding for spacing



# Run the main loop
root.mainloop()
