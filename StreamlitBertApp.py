import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re

# Load the trained model and tokenizer
model_path = "fine_tuned_bert"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cpu")
model.to(device)

# Load the MultiLabelBinarizer classes
categories = ['product_quality', 'customer_service', 'price', 'functionality', 'ease_of_use']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text):
    cleaned_text = clean_text(text)
    encoding = tokenizer(cleaned_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    encoding = {key: value.to(device) for key, value in encoding.items()}
    model.eval()
    with torch.no_grad():
        output = model(**encoding)
        logits = output.logits
        preds = torch.sigmoid(logits).cpu().numpy()
    predictions = (preds >= 0.3).astype(int).flatten()
    return [categories[i] for i, val in enumerate(predictions) if val == 1]

# Streamlit UI
st.title("🔍 Product Review Classifier")
st.markdown("Enter a product review below and get predicted categories:")

user_input = st.text_area("Enter review:", "")

if st.button("Predict"):
    if user_input.strip():
        labels = predict(user_input)
        if labels:
            st.success("Predicted Categories: " + ", ".join(labels))
        else:
            st.warning("No relevant category detected.")
    else:
        st.error("Please enter a review.")
