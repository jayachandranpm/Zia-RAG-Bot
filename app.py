# app.py

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
import secrets
import numpy as np
import re

app = Flask(__name__)

# Generate a secure random key for session management
def generate_secret_key():
    secret_key = secrets.token_hex(16)  # 16 bytes makes a 32-character key
    return secret_key

# Load local documents from JSON file
def load_local_documents(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            local_documents = json.load(f)
        return local_documents
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return []

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to compute BERT embeddings
def compute_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to clean and format text
def clean_and_format_text(text):
    # Remove unnecessary line breaks and extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert plain URLs into clickable hyperlinks
    text = re.sub(r'(https?://\S+)', r'<a href="\1" target="_blank">\1</a>', text)
    
    return text

# Function to perform cosine similarity search using BERT
def perform_search(query, local_documents):
    query_embedding = compute_bert_embeddings(query)
    max_similarity = -1
    best_match = None

    for doc in local_documents:
        doc_text = doc['text']
        doc_embedding = compute_bert_embeddings(doc_text)
        similarity = np.dot(query_embedding, doc_embedding.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = doc
    
    return best_match, max_similarity

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()

        if not user_input:
            return jsonify({'error': 'Please enter a valid question.'})

        # Load local documents
        local_documents = load_local_documents('local_documents.json')

        if not local_documents:
            return jsonify({'error': 'Failed to load local documents.'})

        try:
            # Perform search based on user input
            result_doc, similarity_score = perform_search(user_input, local_documents)

            if not result_doc:
                return jsonify({'error': 'No relevant response found.'})

            # Clean and format the result text
            cleaned_text = clean_and_format_text(result_doc['text'])

            # Prepare HTML response for chatbot interface
            response = f"<div class='message bot-message'>{cleaned_text}</div>"

            return jsonify({'bot_response': response, 'similarity_score': similarity_score})
        
        except Exception as e:
            return jsonify({'error': f'Error processing request: {str(e)}'})

    return jsonify({'error': 'Method not allowed.'})

if __name__ == '__main__':
    # Generate secret key if not set
    if not app.secret_key:
        app.secret_key = generate_secret_key()
        print(f"Generated new secret key: {app.secret_key}")

    app.run(debug=False)
