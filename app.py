from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from docx import Document
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploaded_files'
app.secret_key = 'your_secret_key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load IndoBERT model
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n\n"
    return text

def clean_and_extract_sentences(text):
    paragraphs = text.strip().split("\n\n")
    sentences = []
    for paragraph in paragraphs:
        paragraph = re.sub(r'[^\w\s.,!?]', '', paragraph).strip()
        if len(paragraph.split()) > 3:
            cleaned_paragraph = re.sub(r'\s+', ' ', paragraph.strip())
            potential_sentences = re.split(r'(?<!\w\.\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', cleaned_paragraph)
            for sentence in potential_sentences:
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
    return sentences

def get_sentence_embeddings(sentences, batch_size=16):
    embeddings = []
    model.eval()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def get_key_sentences_kmeans(sentences, embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    key_sentences = []
    for cluster_id in range(num_clusters):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        distances = [(i, np.linalg.norm(embeddings[i] - centers[cluster_id])) for i in indices]
        closest_idx = sorted(distances, key=lambda x: x[1])[0][0]
        key_sentences.append(sentences[closest_idx])

    return key_sentences

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the file
            text = read_docx(filepath)
            sentences = clean_and_extract_sentences(text)
            embeddings = get_sentence_embeddings(sentences)
            key_sentences = get_key_sentences_kmeans(sentences, embeddings, num_clusters=5)
            summary = " ".join(key_sentences)

            return render_template('result.html', sentences=sentences, summary=summary)

    return render_template('index.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
