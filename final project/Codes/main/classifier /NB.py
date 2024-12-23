#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
import random
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics


# In[2]:


AI_dir = r"C:\Users\glend\OneDrive\桌面\MIT_AI"
semi_dir = r"C:\Users\glend\OneDrive\桌面\semiconductor"
ev_dir = r"C:\Users\glend\OneDrive\桌面\ev"
auto_dir = r"C:\Users\glend\OneDrive\桌面\industrial_automation"

all_text = []
all_cls = [] 
content = 'content'

for i in range(1, 1090):
    file_path = os.path.join(AI_dir,content, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    all_text.append(docs_text)
    all_cls.append("AI")


for i in range(1, 2594):
    file_path = os.path.join(semi_dir,content, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    all_text.append(docs_text)
    all_cls.append("semi")
    

for i in range(1, 2437):
    file_path = os.path.join(ev_dir,content, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    all_text.append(docs_text)
    all_cls.append("ev")
    

for i in range(1, 1680):
    file_path = os.path.join(auto_dir,content, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    all_text.append(docs_text)
    all_cls.append("auto")

binary_vectorizer = CountVectorizer(binary=True)
binary_vectors = binary_vectorizer.fit_transform(all_text)

x_train, x_test, y_train, y_test = train_test_split(
    binary_vectors, all_cls, test_size=0.1, stratify=all_cls, random_state=42
)


model = BernoulliNB()
model.fit(x_train, y_train)

predicted_results = model.predict(x_test)

print(metrics.classification_report(y_test, predicted_results))




# In[3]:


score = model.predict_proba(x_test)  
n_classes = len(model.classes_)

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test, score[:, i], pos_label=model.classes_[i])
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, lw=2, label=f'Class {model.classes_[i]} (AUC = {auc_score:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision-Recall Curve for each class")
plt.show()


# In[ ]:




