import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
# import nltk
# nltk.download('stopwords')

current_dir = os.getcwd()
ev_dir = 'ev'
content = 'content'
date = 'date'
docs_dir_ev = os.path.join(current_dir, ev_dir, content)
dates_dir_ev = os.path.join(current_dir, ev_dir, date)

all_text = []
all_cls = []  # List to store class labels
all_dates = []

# Initialize the NLTK stopwords
# stop_words = set(nltk.corpus.stopwords.words('english'))

# def preprocess_text(text):
#     # Convert text to lowercase
#     text = text.lower()
#     # Split the text into words
#     words = text.split()
#     # Remove stopwords
#     words = [word for word in words if word not in stop_words]
#     # Rejoin the words into a single string
#     return " ".join(words)

for i in range(1, 2437):  # Adjust the range if needed
    file_path = os.path.join(docs_dir_ev, f"{i}.txt")
    date_file_path = os.path.join(dates_dir_ev, f"{i}.txt")

    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()

    with open(date_file_path, "r", encoding="utf-8") as date_file:
        date_str = date_file.read()

    # Convert date_str to datetime
    date = pd.to_datetime(date_str, errors='coerce')

    if date is not None and date.year >= 2022:
        # docs_text = preprocess_text(docs_text)
        all_dates.append(date)
        all_text.append(docs_text)
        all_cls.append("AI")

TF_vectorizer = CountVectorizer(min_df=1, stop_words='english')
TF_vectors = TF_vectorizer.fit_transform(all_text)

# Use Latent Dirichlet Allocation (LDA) instead of TruncatedSVD
n_topics = 10  # Choose the number of topics
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
LDA_vectors = lda_model.fit_transform(TF_vectors)

def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names_out()
    for topic_index, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_index)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_topics(lda_model, TF_vectorizer, 10)

# Create DataFrame for analysis
df = pd.DataFrame({'Season': [date.quarter for date in all_dates],
                   'Year': [date.year for date in all_dates],
                   'Topic': LDA_vectors.argmax(axis=1)})

# Group by season, year, and topic, count the number of documents
result = df.groupby(['Season', 'Year', 'Topic']).size().reset_index(name='Count')

# Pivot the result DataFrame
result_pivoted = result.pivot_table(index=['Year', 'Season'], columns='Topic', values='Count', fill_value=0).astype(int)

# Print the pivoted result
print(result_pivoted)
