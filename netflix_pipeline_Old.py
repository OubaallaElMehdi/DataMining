import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ================== 1. Load Data ====================
df = pd.read_csv("netflix_titles.csv")  # Ensure the CSV is in the same folder.

# We’ll classify whether it's a Movie or TV Show, so let's drop empty descriptions
df.dropna(subset=['description'], inplace=True)

# ================== 2. Preprocessing =================
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words & non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df['clean_description'] = df['description'].apply(preprocess_text)

# ================== 3. TF-IDF Extraction =================
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['clean_description'])

# ================== 4. Label =================
# We'll predict if it's 'Movie' or 'TV Show'
y = df['type']

# ================== 5. Firefly Feature Selection (Optional) =================
# This is a placeholder. If you have a Firefly method, call it here.
"""
def firefly_feature_selection(X, y):
    # You would implement or call the Firefly algorithm.
    # Return indices for the best features.
    return range(X.shape[1])  # Fake: returning all features

selected_indices = firefly_feature_selection(X_tfidf.toarray(), y)
X_final = X_tfidf[:, selected_indices]
"""

# If not using Firefly, just do:
X_final = X_tfidf

# ================== 6. Train/Test Split =================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# ================== 7. Classification Examples =================

# --- A) Multinomial Naive Bayes ---
mnb_clf = MultinomialNB()
mnb_clf.fit(X_train, y_train)
mnb_pred = mnb_clf.predict(X_test)

mnb_acc = accuracy_score(y_test, mnb_pred)
mnb_prec = precision_score(y_test, mnb_pred, average='macro')
mnb_rec = recall_score(y_test, mnb_pred, average='macro')
mnb_f1 = f1_score(y_test, mnb_pred, average='macro')

print("=== Multinomial NB Results ===")
print(f"Accuracy:  {mnb_acc:.3f}")
print(f"Precision: {mnb_prec:.3f}")
print(f"Recall:    {mnb_rec:.3f}")
print(f"F1 Score:  {mnb_f1:.3f}\n")

# --- B) Complement Naive Bayes ---
cnb_clf = ComplementNB()
cnb_clf.fit(X_train, y_train)
cnb_pred = cnb_clf.predict(X_test)

cnb_acc = accuracy_score(y_test, cnb_pred)
cnb_prec = precision_score(y_test, cnb_pred, average='macro')
cnb_rec = recall_score(y_test, cnb_pred, average='macro')
cnb_f1 = f1_score(y_test, cnb_pred, average='macro')

print("=== Complement NB Results ===")
print(f"Accuracy:  {cnb_acc:.3f}")
print(f"Precision: {cnb_prec:.3f}")
print(f"Recall:    {cnb_rec:.3f}")
print(f"F1 Score:  {cnb_f1:.3f}\n")

# --- C) SVM ---
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, average='macro')
svm_rec = recall_score(y_test, svm_pred, average='macro')
svm_f1 = f1_score(y_test, svm_pred, average='macro')

print("=== SVM Results ===")
print(f"Accuracy:  {svm_acc:.3f}")
print(f"Precision: {svm_prec:.3f}")
print(f"Recall:    {svm_rec:.3f}")
print(f"F1 Score:  {svm_f1:.3f}")
