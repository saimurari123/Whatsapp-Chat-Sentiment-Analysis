import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib 

def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True):

    normalized_corpus = []

    for doc in corpus:

        if html_stripping:
            doc = strip_html_tags(doc)

        if accented_char_removal:
            doc = remove_accented_chars(doc)

        if contraction_expansion:
            doc = expand_contractions(doc)

        if text_lower_case:
            doc = doc.lower()


        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)

        if text_lemmatization:
            doc = lemmatize_text(doc)

        if special_char_removal:
            doc = remove_special_characters(doc)


        doc = re.sub(' +', ' ', doc)

        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

    return normalized_corpus

model_file = 'C:\Users\saimu\Documents\Ml_lab\ml_project\web_app\logistic_model.pkl'  # Placeholder for actual model
vectorizer_file = 'C:\Users\saimu\Documents\Ml_lab\ml_project\web_app\vectorizer.pkl'  # Placeholder for actual vectorizer

loaded_model = joblib.load(model_file) 
loaded_vectorizer = joblib.load(vectorizer_file) 


def analyze_chat(file_path, model, vectorizer):
    opinion = {}
    messages = []
    pos, neg = 0, 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                name_msg = line.split('-')[1]
                name = name_msg.split(':')[0].strip()
                chat = name_msg.split(':', 1)[1].strip()
                messages.append((name, chat))

                
                if opinion.get(name) is None:
                    opinion[name] = [0, 0]  

                
                normalized = normalize_corpus([chat])
                vectorized = vectorizer.transform(normalized)
                res = model.predict(vectorized)[0]

                if res == 'positive':
                    pos += 1
                    opinion[name][0] += 1
                else:
                    neg += 1
                    opinion[name][1] += 1

            except Exception as e:
                continue

    return opinion, pos, neg, messages



st.title("WhatsApp Chat Sentiment Analysis")
st.sidebar.header("Upload Chat File")
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file", type="txt")

if uploaded_file is not None:
    
    st.write(f"**Analyzing file**: {uploaded_file.name}")

    
    opinion, pos, neg, messages = analyze_chat(uploaded_file, loaded_model, loaded_vectorizer)

    # Display sentiment count
    st.write(f"**Total Positive Messages: {pos}**")
    st.write(f"**Total Negative Messages: {neg}**")

    # Display summary of sentiments per person
    st.subheader("Sentiment Summary per Person")
    for name, (p, n) in opinion.items():
        st.write(f"👤 {name} → 😊 Positive: {p}, 😞 Negative: {n}")

    # Plotting the sentiment count per person
    names = list(opinion.keys())
    positive_counts = [opinion[n][0] for n in names]
    negative_counts = [opinion[n][1] for n in names]

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(names))
    ax.bar(x, positive_counts, width=0.4, label='Positive 😊', color='green', align='center')
    ax.bar(x, negative_counts, width=0.4, label='Negative 😞', color='red', align='edge')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set_xlabel("Person")
    ax.set_ylabel("Message Count")
    ax.set_title("Sentiment Count Per Person (WhatsApp Chat)")
    ax.legend()

    # Display the plot
    st.pyplot(fig)

else:
    st.write("Please upload a WhatsApp chat file to get started.")

