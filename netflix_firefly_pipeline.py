import pandas as pd
import numpy as np
import joblib  # For saving/loading models
import time
import os

# Scikit-learn
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import SVC

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For plotting
import matplotlib.pyplot as plt

##########################################################
# 1) LOAD DATA
##########################################################
df = pd.read_csv("netflix_titles.csv")  # Must be in the same folder

# Drop rows with missing descriptions
df.dropna(subset=['description'], inplace=True)

# We'll classify whether it's a 'Movie' or a 'TV Show'
X_raw = df['description']
y = df['type']

##########################################################
# 2) PREPROCESSING (Tokenization + Stop Word Removal)
##########################################################
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove non-alphabetic tokens + stop words
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

X_processed = X_raw.apply(preprocess_text)

##########################################################
# 3) FEATURE EXTRACTION (TF-IDF) 
##########################################################
tfidf = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf.fit_transform(X_processed)

##########################################################
# 4) FEATURE SELECTION (FIRELY ALGORITHM) - Placeholder
##########################################################
def firefly_feature_selection(X, y):
    return np.arange(X.shape[1])

selected_indices = firefly_feature_selection(X_tfidf, y)
X_selected = X_tfidf[:, selected_indices]

##########################################################
# 5) TRAIN/TEST SPLIT (For saving .pkl models)
##########################################################
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

##########################################################
# 6) FUNCTION TO TRAIN & EVALUATE A PIPELINE
##########################################################
def run_pipeline(classifier, model_name, save_filename):
    print(f"=== {model_name} ===")

    start_time = time.time()
    classifier.fit(X_train.toarray(), y_train)  # Some require dense
    end_time = time.time()

    y_pred = classifier.predict(X_test.toarray())

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy:   {acc:.3f}")
    print(f"Precision:  {prec:.3f}")
    print(f"Recall:     {rec:.3f}")
    print(f"F1 Score:   {f1:.3f}")
    print(f"Train Time: {end_time - start_time:.2f}s\n")

    # === Save the trained model ===
    joblib.dump(classifier, save_filename)
    if os.path.exists(save_filename):
        print(f"Model saved to {save_filename}\n")
    else:
        print(f"Error saving model {save_filename}\n")

##########################################################
# 7) RUN EACH CLASSIFIER ONCE (train/test) + SAVE
##########################################################
# (A) GaussianNB
gnb = GaussianNB()
run_pipeline(
    classifier=gnb,
    model_name="Naïve Bayes (GaussianNB)",
    save_filename="gaussian_nb_model.pkl"
)

# (B) SVM (kernel='linear', verbose for progress)
svm_clf = SVC(kernel='linear', verbose=True)
run_pipeline(
    classifier=svm_clf,
    model_name="SVM (linear)",
    save_filename="svm_model.pkl"
)

# (C) MultinomialNB
mnb_clf = MultinomialNB()
run_pipeline(
    classifier=mnb_clf,
    model_name="Multinomial Naïve Bayes",
    save_filename="multinomial_nb_model.pkl"
)

# (D) ComplementNB
cnb_clf = ComplementNB()
run_pipeline(
    classifier=cnb_clf,
    model_name="Complement Naïve Bayes",
    save_filename="complement_nb_model.pkl"
)

##########################################################
# 8) PLOT SEPARATE LEARNING CURVES (ONE FOR EACH CLASSIFIER)
##########################################################

def plot_learning_curve_single(clf, clf_name, X, y):
    print(f"Generating learning curve for: {clf_name}")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=clf,
        X=X.toarray(),
        y=y,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    # Calculate mean & std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train Accuracy')
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.1, color='blue')

    plt.plot(train_sizes, test_mean, 'o--', color='orange', label='Validation Accuracy')
    plt.fill_between(train_sizes,
                     test_mean - test_std,
                     test_mean + test_std,
                     alpha=0.1, color='orange')

    plt.title(f"Learning Curve - {clf_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(loc="best")

    # Save figure
    filename_fig = f"{clf_name}_learning_curve.png"
    plt.savefig(filename_fig)
    plt.close()  # Close the figure to free memory
    print(f"Saved learning curve to {filename_fig}\n")

# We now call the function for each of the 4 classifiers
# We pass in X_selected, y (the entire dataset)
plot_learning_curve_single(gnb, "GaussianNB", X_selected, y)
plot_learning_curve_single(svm_clf, "SVM_linear", X_selected, y)
plot_learning_curve_single(mnb_clf, "MultinomialNB", X_selected, y)
plot_learning_curve_single(cnb_clf, "ComplementNB", X_selected, y)
