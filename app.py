import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import networkx as nx
import torch
import numpy as np
import spacy
import sentencepiece  # Ensure sentencepiece is installed

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load("en_core_sci_lg")

# Load the pre-trained BERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load Hugging Face T5 model for abstractive summarization
t5_model_name = "t5-base"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

# Preprocess text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

# Get sentence embeddings
def get_sentence_embeddings(sentences, model, tokenizer):
    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
            embeddings.append(sentence_embedding.squeeze().numpy())
    return np.array(embeddings)

# Build semantic graph
def build_semantic_graph(embeddings, similarity_threshold=0.75):
    graph = nx.Graph()
    for i, emb1 in enumerate(embeddings):
        for j, emb2 in enumerate(embeddings):
            if i != j:
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if similarity >= similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)
    return graph

# Apply TextRank
def apply_textrank(graph, sentences, damping_factor=0.85, max_iter=100):
    num_nodes = len(sentences)
    personalization = {i: 1 / num_nodes for i in range(num_nodes)}
    scores = nx.pagerank(graph, personalization=personalization, max_iter=max_iter)
    ranked_sentences = sorted(((score, idx) for idx, score in scores.items()), reverse=True)
    return ranked_sentences

# Generate summary
def generate_summary(ranked_sentences, sentences, max_length_ratio=0.5):
    stop_words = set(stopwords.words('english'))
    summary = []
    current_length = 0
    total_length = sum(len(sentence.split()) for sentence in sentences)
    max_length = int(total_length * max_length_ratio)

    for score, idx in ranked_sentences:
        sentence = sentences[idx]
        sentence_length = len(sentence.split())
        sentence_words = [word for word in sentence.split() if word.lower() not in stop_words]

        if current_length + sentence_length <= max_length:
            summary.append(" ".join(sentence_words))
            current_length += sentence_length
        else:
            break

    return " ".join(summary)

# Extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Perform abstractive summarization
def abstractive_summary(text, max_length_ratio=0.2, min_length_ratio=0.1):
    total_length = len(text.split())
    max_length = int(total_length * max_length_ratio)
    min_length = int(total_length * min_length_ratio)
    
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app layout
st.title("Hybrid Summarization App")
st.write("Choose between single-document and multi-document summarization.")

# User choice for summarization type
summarization_type = st.radio("Select Summarization Type", ("Single-Document", "Multi-Document"))

if summarization_type == "Multi-Document":
    st.header("Multi-Document Summarization")
    uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)

    if uploaded_files:
        texts = [file.read().decode("utf-8") for file in uploaded_files]
        
        # Perform extractive summarization for each document
        extractive_summaries = []
        for text in texts:
            sentences = preprocess_text(text)
            embeddings = get_sentence_embeddings(sentences, model, tokenizer)
            graph = build_semantic_graph(embeddings)
            ranked_sentences = apply_textrank(graph, sentences)
            ext_summary = generate_summary(ranked_sentences, sentences, max_length_ratio=0.5)
            extractive_summaries.append(ext_summary)
        
        # Combine extractive summaries for multi-document summarization
        combined_extractive_summary = " ".join(extractive_summaries)
        
        
        # Extract named entities from the combined summary
        entities = extract_named_entities(combined_extractive_summary)
        
        
        # Choose summary length ratio for abstractive summarization
        abs_ratio_option = st.selectbox("Choose abstractive summary length ratio", ("1/2", "1/4"))
        abs_ratio = {"1/2": 0.5,  "1/4": 0.25}[abs_ratio_option]

        # Combine extractive summary and named entities
        combined_input = combined_extractive_summary + " Keywords: " + ', '.join([ent[0] for ent in entities])

        # Perform abstractive summarization
        abs_summary = abstractive_summary(combined_input, max_length_ratio=abs_ratio, min_length_ratio=abs_ratio/2)
        st.write("Abstractive Summary:", abs_summary)

elif summarization_type == "Single-Document":
    st.header("Single-Document Summarization")

    # Add file uploader
    uploaded_file = st.file_uploader("Upload a text file", type="txt")

    # Add text area for manual input
    text_input = st.text_area("Or enter text here")

    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")

    if text_input:
        # Extract named entities
        entities = extract_named_entities(text_input)
        

        # Perform extractive summarization
        sentences = preprocess_text(text_input)
        embeddings = get_sentence_embeddings(sentences, model, tokenizer)
        graph = build_semantic_graph(embeddings)
        ranked_sentences = apply_textrank(graph, sentences)
        ext_summary = generate_summary(ranked_sentences, sentences, max_length_ratio=0.5)
        

        # Choose summary length ratio for abstractive summarization
        abs_ratio_option = st.selectbox("Choose abstractive summary length ratio", ("1/2",  "1/4"))
        abs_ratio = {"1/2": 0.5, "1/4": 0.25}[abs_ratio_option]

        # Combine extractive summary and named entities
        combined_input = ext_summary + " Keywords: " + ', '.join([ent[0] for ent in entities])

        # Perform abstractive summarization
        abs_summary = abstractive_summary(combined_input, max_length_ratio=abs_ratio, min_length_ratio=abs_ratio/2)
        st.write("Abstractive Summary:", abs_summary)
