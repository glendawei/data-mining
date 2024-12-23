#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
import nltk
import random
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import pandas as pd

direction = r"C:\Users\glend\OneDrive\桌面\semiconductor"
all_text = []
all_cls = []  
content_dir = os.path.join(direction, "content")
date_dir = os.path.join(direction, "date")
all_dates = []




for i in range(1, 2594):
    file_path = os.path.join(content_dir, f"{i}.txt")
    date_file_path = os.path.join(date_dir, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    with open(date_file_path, "r", encoding="utf-8") as date_file:
        date_str = date_file.read()
    all_dates.append(pd.to_datetime(date_str, errors='coerce'))
    all_text.append(docs_text)
    all_cls.append("ev")
    

    

TFIDF_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
TFIDF_vectors = TFIDF_vectorizer.fit_transform(all_text)


svd_model = TruncatedSVD(n_components = 10)
SVD_vectors = svd_model.fit_transform(TFIDF_vectors)
SVD_vectors.shape
svd_model.components_.shape
svd_model.singular_values_

def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names_out()
    for topic_index, topic in enumerate(abs(model.components_)):
        print("\nTopic #%d:" % topic_index)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_topics(svd_model, TFIDF_vectorizer, 10)


# In[2]:


df = pd.DataFrame({'Season': [date.quarter for date in all_dates], 'Year': [date.year for date in all_dates], 'Topic': SVD_vectors.argmax(axis=1)})

# Group by season, year, and topic, count the number of documents
result = df.groupby(['Season', 'Year', 'Topic']).size().reset_index(name='Count')

# Pivot the result DataFrame
result_pivoted = result.pivot_table(index=['Year', 'Season'], columns='Topic', values='Count', fill_value=0).astype(int)

# Print the pivoted result
print(result_pivoted)


# In[ ]:




