from preprocess import CATEGORY_INVERSED
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import pickle

reviews = [
    "Great movie",
    "Aside from bad ending, I really like this movie",
    "When the next season will be out?",
    "Totally mid", #using some slang and new word
    "Bad movie",
    "Cannot admit more, this product can't be better then the previous one",
    "Really good and bad movie" # makes no sense
]

print("Loading counting vector...")
count_vector = pickle.load(open("results/count_vector.pickle", "rb"))

print("Loading model...")
naive_bayes = pickle.load(open("results/naive_bayes.pickle", "rb"))

x = count_vector.transform(reviews)
result = naive_bayes.predict(x)
result = [ CATEGORY_INVERSED[res] for res in result ]
print(result)