import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
import numpy as np
import nltk
import random

current_dir = os.getcwd()
AI_dir = 'MIT_AI'
semi_dir = 'semiconductor'
ev_dir = 'ev'
auto_dir = 'industrial_automation'
content = 'content'
docs_dir_AI = os.path.join(current_dir, AI_dir, content)
docs_dir_semi = os.path.join(current_dir, semi_dir, content)
docs_dir_ev = os.path.join(current_dir, ev_dir, content)
docs_dir_auto = os.path.join(current_dir, auto_dir, content)

AI_rand = [random.randint(1,1089) for _ in range(250)] 
semi_rand = [random.randint(1,2593) for _ in range(250)] 
ev_rand = [random.randint(1,2436) for _ in range(250)] 
auto_rand = [random.randint(1,1679) for _ in range(250)] 

# /////////////Preprocess/////////////////////
def preprocess_text(text):
    # Convert text to lowercase (case folding)
    text = text.lower()
    # Split the text into words
    words = nltk.word_tokenize(text)
    
    return " ".join(words)

# //////////////取資料////////////////
docs = []
doc_ids = []

for i in AI_rand:
    # Use the with statement to open and read each file
    file_path = os.path.join(docs_dir_AI, f"{i}.txt")
    # Check if the file exists
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    # Preprocess the text and add it to the list of documents
    docs_text = preprocess_text(docs_text)
    docs.append(docs_text)
    doc_ids.append("1")

for i in semi_rand:
    # Use the with statement to open and read each file
    file_path = os.path.join(docs_dir_semi, f"{i}.txt")
    # Check if the file exists
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    # Preprocess the text and add it to the list of documents
    docs_text = preprocess_text(docs_text)
    docs.append(docs_text)
    doc_ids.append("2")
    
for i in ev_rand:
    # Use the with statement to open and read each file
    file_path = os.path.join(docs_dir_ev, f"{i}.txt")
    # Check if the file exists
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    # Preprocess the text and add it to the list of documents
    docs_text = preprocess_text(docs_text)
    docs.append(docs_text)
    doc_ids.append("3")
    
for i in auto_rand:
    # Use the with statement to open and read each file
    file_path = os.path.join(docs_dir_auto, f"{i}.txt")
    # Check if the file exists
    with open(file_path, "r", encoding="utf-8") as file:
        docs_text = file.read()
    # Preprocess the text and add it to the list of documents
    docs_text = preprocess_text(docs_text)
    docs.append(docs_text)
    doc_ids.append("4")

# //////////////////Tfidf vectors //////////////////
vectorizer = TfidfVectorizer()
tfidf_docs = vectorizer.fit_transform(docs)

# //////////////////evaluation values///////////////////
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import matplotlib.pyplot as plt

train, test, train_doc_ids, test_doc_ids = train_test_split(tfidf_docs, doc_ids, test_size=0.1, random_state=0, stratify=doc_ids)

model_svm_linear = SVC(kernel='linear', C=1.0)
model_svm_linear.fit(train, train_doc_ids)

# Binarize the labels
eval_bin = label_binarize(test_doc_ids, classes=np.unique(doc_ids))

predicted_result=[]
expected_result=[]

predicted_result.extend(model_svm_linear.predict(test))
expected_result.extend(test_doc_ids)

print(metrics.classification_report(expected_result, predicted_result))

# Compute decision function scores
score = model_svm_linear.decision_function(test)

# Plot the precision-recall curve for each class
n_classes = eval_bin.shape[1]

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(eval_bin[:, i], score[:, i])
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, lw=2, label=f'Class {i+1} (AUC = {auc_score:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision-Recall Curve for each class")
plt.show()