# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZLHV4UxkhN26RWcI2LTKmLP4BL82UPT6

**Write a program to convert a set of documents into TF-IDF vectors**

  ○ Text collection: 1095 news documents

  ○ 1.txt ~ 1095.txt

  ○ Use TfidfVectorizer

    ■ Lowercase everything

    ■ Filter out English stopwords
"""

from google.colab import drive
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

drive.mount('/content/drive')

document_folder = "/content/drive/My Drive/PA1-data"

documents = []


for i in range(1, 1096):
    with open(os.path.join(document_folder, f"{i}.txt"), 'r', encoding='utf-8') as file:
        content = file.read()
        documents.append(content)

tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

"""**Save each document’s TF-IDF vector as a plain text file**

○ 1.vec, 2.vec, ... 1095.vec
"""

output_folder = "/content/drive/My Drive/TF-IDF-Vectors"

os.makedirs(output_folder, exist_ok=True)

for i in range(1095):
    filename = os.path.join(output_folder, f"{i + 1}.vec")
    with open(filename, 'w') as file:
        for item in tfidf_matrix[i].toarray()[0]:
            file.write(str(item) + '\n')

"""**Load the TF-IDF vectors of documents 1 and 2, and calculate their cosine similarity.**"""

import numpy as np

matr1 = np.loadtxt("/content/drive/My Drive/TF-IDF-Vectors/1.vec")
matr2 = np.loadtxt("/content/drive/My Drive/TF-IDF-Vectors/2.vec")
matr1 = matr1.reshape(1, -1)
matr2 = matr2.reshape(1, -1)
cosine_sim = cosine_similarity(matr1 , matr2 ).flatten()[0]
print(f"Cosine Similarity between Document 1 and Document 2: {cosine_sim}")