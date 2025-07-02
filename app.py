import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import git

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_essay(essay):
    essay = essay.lower()
    essay = re.sub(r'[^a-z\s]', '', essay)
    return ' '.join([word for word in essay.split() if word not in stop_words])


def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    svm = joblib.load("models/svm_model.pkl")
    tree = joblib.load("models/decision_tree_model.pkl")
    ada = joblib.load("models/adaboost_model.pkl")

    return tfidf, svm, tree, ada

tfidf, svm_mod, tree_mod, ada_mod = load_models()


st.title("AI vs Human Text Identifier")
st.markdown("Enter a essay and choose a model to identify if the text is AI or Human crafted")

user_input = st.text_area("Enter essay", height=200)

model_choice = st.selectbox("Choose a model", ("SVM", "Decision Tree", "AdaBoost"))

if st.button("Identify"):
    if user_input.strip() == "":
        st.warning("Please enter a essay to be identified")
    else:
        valid_input = clean_essay(user_input)
        vectored_input = tfidf.transform([valid_input])

        if model_choice == "SVM":
            pred = svm_mod.predict(vectored_input)[0]
            prob = svm_mod.predict_proba(vectored_input)[0][pred]
            st.markdown("The SVM Classifier excels at finding the optimal plane to separate data points or in this case words to identify differences. It takes apart the essay systematically to identify aspects that the data " \
                "set has determined to be the differences between AI and Human writing. It's margins give it near 90 percent accuracy in determining these differences.")
        elif model_choice == "Decision Tree":
            pred = tree_mod.predict(vectored_input)[0]
            prob = tree_mod.predict_proba(vectored_input)[0][pred]
            st.markdown("Decision Tree Classifier takes apart the essay via predetermined feature values, determining the differences of what Humans and AI view as important in sentence structure. The key working of the Decision Tree breaks down the essay into nodes and branches creating a tree or roots that identify the important break points in the feature values. The accuracy of Decision Tree come to around 87 percent.")
        elif model_choice == "AdaBoost":
            pred = ada_mod.predict(vectored_input)[0]
            prob = ada_mod.predict_proba(vectored_input)[0][pred]
            st.markdown("The AdaBoost Classifier is an ensemble learning method, combining multiple learning models into one stronger classifier. It focuses more on the incorrect data, tuning itself to recognize the oddities until its accuracy reaches a level determined by the researcher. The accuracy level was set to 90 percent for the sake of this project.")
        
        label = "Human" if pred == 0 else "AI"
        st.success(f"Prediction: {label}")