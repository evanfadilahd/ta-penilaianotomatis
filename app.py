from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
from docx import Document
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import sqlite3
from functools import wraps
import json
import pdfplumber

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploaded_files'
app.secret_key = 'sk_ta_evan'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load IndoBERT model
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_db_connection():
    conn = sqlite3.connect('projek_skripsi.db')
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'lecturer_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def read_docx(file_path):
    """Read text from a .docx file using python-docx"""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n\n"
    return text

def read_pdf(file_path):
    """Read text from a PDF file using pdfplumber"""
    with pdfplumber.open(file_path) as pdf:
        text = "\n\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def clean_and_extract_sentences(text):
    """Clean text and extract sentences using regex"""
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
            # sentences += [s.strip() for s in potential if s.strip()]
    return sentences

def get_sentence_embeddings(sentences, batch_size=16):
    """Generate embeddings for sentences using IndoBERT"""
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

def determine_optimal_clusters(embeddings, candidate_clusters=[3, 5, 7, 9, 11, 13]):
    best_k = candidate_clusters[0]
    best_score = -1
    scores = {}
    for k in candidate_clusters:
        if k >= len(embeddings):
            continue
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores

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
        key_sentences.append((closest_idx, sentences[closest_idx]))

    return [s for i, s in sorted(key_sentences, key=lambda x: x[0])], labels, centers

def divide_sentences_by_index(sentences, num_clusters):
    num_sentences = len(sentences)
    sentences_per_cluster = num_sentences // num_clusters
    clusters = [
        i // sentences_per_cluster for i in range(num_sentences)
    ]
    return clusters

def compute_rouge(reference_summary, generated_summary):
    """Calculate ROUGE scores using rouge-score"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference_summary, generated_summary)

def process_with_k(sentences, embeddings, k):
    """Process text with a specific k value for clustering"""
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Get key sentences
    key_sentences = []
    for cluster_id in range(k):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        distances = [(i, np.linalg.norm(embeddings[i] - centers[cluster_id])) for i in indices]
        closest_idx = sorted(distances, key=lambda x: x[1])[0][0]
        key_sentences.append((closest_idx, sentences[closest_idx]))

    # Calculate silhouette score
    sil_score = silhouette_score(embeddings, labels)
    
    # Generate summary
    final_summary = " ".join([s for i, s in sorted(key_sentences, key=lambda x: x[0])])
    
    return {
        'num_clusters': k,
        'silhouette_score': sil_score,
        'cluster_labels': labels.tolist(),
        'cluster_centers': centers.tolist(),
        'key_sentences': [s for i, s in sorted(key_sentences, key=lambda x: x[0])],
        'final_summary': final_summary
    }

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        lecturer = conn.execute('SELECT * FROM Lecturer WHERE lecturer_email = ? AND lecturer_password = ?', (email, password)).fetchone()
        if lecturer:
            session['lecturer_id'] = lecturer['lecturer_id']
            session['lecturer_name'] = lecturer['lecturer_name']
            # Log activity
            conn.execute('INSERT INTO Activity_Log (lecturer_id, activity, created_at) VALUES (?, ?, datetime("now"))', (lecturer['lecturer_id'], 'login',))
            conn.commit()
            conn.close()
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
            conn.close()
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    return render_template('dashboard.html', user_name=session.get('lecturer_name', ''))

@app.route('/upload', methods=['GET'])
@login_required
def upload():
    return render_template('upload_tugas_baru.html', user_name=session.get('lecturer_name', ''))

@app.route('/history', methods=['GET'])
@login_required
def history():
    conn = get_db_connection()
    # Join Student_Assignment, Summary, and Answer_Key
    rows = conn.execute('''
        SELECT sa.assignment_id, sa.student_name, sa.assignment, sa.uploaded_at, s.summary_text, s.summary_file, ak.text_key
        FROM Student_Assignment sa
        JOIN Summary s ON sa.assignment_id = s.assignment_id
        JOIN Answer_Key ak ON sa.lecturer_id = ak.lecturer_id
        ORDER BY sa.uploaded_at DESC
    ''').fetchall()
    conn.close()
    # Prepare data for template
    summaries = []
    for row in rows:
        summary = row['summary_text']
        answer_key = row['text_key']
        percent_sim, grade, value = compute_grading(summary, answer_key)
        summaries.append({
            'student_name': row['student_name'],
            'assignment': row['assignment'],
            'uploaded_at': row['uploaded_at'],
            'summary': summary,
            'summary_file': row['summary_file'],
            'grading': {
                'percent_sim': float(percent_sim),
                'grade': grade,
                'value': int(value)
            }
        })
    return render_template('hasil_ringkasan.html', user_name=session.get('lecturer_name', ''), summaries=summaries)

@app.route('/statistic', methods=['GET'])
@login_required
def statistic():
    try:
        conn = get_db_connection()
        rows = conn.execute('''
            SELECT 
                sa.assignment_id, sa.student_name, sa.assignment, sa.uploaded_at, 
                s.summary_text, s.summary_file, ak.text_key,
                ps.cleaned_text, ps.extracted_sentences, ps.embeddings,
                ps.silhouette_score, ps.num_clusters, ps.cluster_labels,
                ps.key_sentences, ps.final_summary
            FROM Student_Assignment sa
            JOIN Summary s ON sa.assignment_id = s.assignment_id
            JOIN Answer_Key ak ON sa.lecturer_id = ak.lecturer_id
            JOIN Processing_Steps ps ON sa.assignment_id = ps.assignment_id
            ORDER BY sa.uploaded_at DESC
        ''').fetchall()
        conn.close()
        
        # Prepare data for template
        stats = []
        for row in rows:
            try:
                summary = row['summary_text']
                answer_key = row['text_key']
                percent_sim, grade, value = compute_grading(summary, answer_key)
                
                # Get processing steps data
                extracted_sentences = json.loads(row['extracted_sentences'])
                embeddings = json.loads(row['embeddings'])
                cluster_labels = json.loads(row['cluster_labels'])
                key_sentences = json.loads(row['key_sentences'])
                
                # Calculate silhouette scores for different k values
                silhouette_scores = {}
                candidate_clusters = [3, 5, 7, 9, 11, 13]
                for k in candidate_clusters:
                    if k >= len(embeddings):
                        continue
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    silhouette_scores[k] = float(score)
                
                # Group sentences by cluster
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(extracted_sentences[i])
                
                # Calculate ROUGE scores
                rouge_scores = compute_rouge(answer_key, summary)
                
                stats.append({
                    'student_name': row['student_name'],
                    'assignment': row['assignment'],
                    'uploaded_at': row['uploaded_at'],
                    'summary': summary,
                    'summary_file': row['summary_file'],
                    'grading': {
                        'percent_sim': float(percent_sim),
                        'grade': grade,
                        'value': int(value)
                    },
                    'processing': {
                        'sentences': extracted_sentences,
                        'embeddings_shape': [int(x) for x in np.array(embeddings).shape],
                        'embeddings': embeddings,
                        'silhouette_scores': silhouette_scores,
                        'clusters': clusters,
                        'key_sentences': key_sentences,
                        'num_clusters': int(row['num_clusters']),
                        'rouge_scores': {
                            'rouge1': {
                                'precision': float(rouge_scores['rouge1'].precision),
                                'recall': float(rouge_scores['rouge1'].recall),
                                'fmeasure': float(rouge_scores['rouge1'].fmeasure)
                            },
                            'rouge2': {
                                'precision': float(rouge_scores['rouge2'].precision),
                                'recall': float(rouge_scores['rouge2'].recall),
                                'fmeasure': float(rouge_scores['rouge2'].fmeasure)
                            },
                            'rougeL': {
                                'precision': float(rouge_scores['rougeL'].precision),
                                'recall': float(rouge_scores['rougeL'].recall),
                                'fmeasure': float(rouge_scores['rougeL'].fmeasure)
                            }
                        }
                    }
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        return render_template('statistika.html', user_name=session.get('lecturer_name', ''), stats=stats, zip=zip)
    except Exception as e:
        print(f"Error in statistic route: {e}")
        flash('An error occurred while processing statistics')
        return redirect(url_for('dashboard'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_post():
    name = request.form['nama-tugas']
    answer_key = request.form['kunci-jawaban']
    student_files = request.files.getlist('tugas-mahasiswa')
    summary_files = request.files.getlist('ringkasan-mahasiswa')

    conn = get_db_connection()
    lecturer_id = session['lecturer_id']

    def extract_sentences(file_path):
        """Extract sentences from file based on its type"""
        if file_path.endswith('.docx'):
            text = read_docx(file_path)
        elif file_path.endswith('.pdf'):
            text = read_pdf(file_path)
        else:
            text = ''
        return text, clean_and_extract_sentences(text)

    # Save all student assignment files and process each
    for idx, student_file in enumerate(student_files):
        student_filename = secure_filename(student_file.filename)
        student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
        student_file.save(student_path)
        
        # 1. Text Cleaning
        cleaned_text, sentences = extract_sentences(student_path)
        
        # 2. Sentence Extraction
        extracted_sentences = sentences
        
        # 3. Generate Embeddings
        embeddings = get_sentence_embeddings(sentences)
        
        # 4. Process with different k values
        candidate_clusters = [3, 5, 7, 9, 11, 13]
        processing_results = {}
        best_k = None
        best_score = -1
        
        for k in candidate_clusters:
            if k >= len(embeddings):
                continue
            result = process_with_k(sentences, embeddings, k)
            processing_results[k] = result
            if result['silhouette_score'] > best_score:
                best_score = result['silhouette_score']
                best_k = k
        
        # Save assignment
        conn.execute('INSERT INTO Student_Assignment (lecturer_id, student_name, assignment, uploaded_at) VALUES (?, ?, ?, datetime("now"))',
                     (lecturer_id, name, student_filename))
        assignment_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        
        # Save processing steps for each k value
        for k, result in processing_results.items():
            conn.execute('''
                INSERT INTO Processing_Steps (
                    assignment_id, cleaned_text, extracted_sentences, embeddings,
                    silhouette_score, num_clusters, cluster_labels, key_sentences,
                    final_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))
            ''', (
                assignment_id,
                cleaned_text,
                json.dumps(extracted_sentences),
                json.dumps(embeddings.tolist()),
                result['silhouette_score'],
                k,
                json.dumps(result['cluster_labels']),
                json.dumps(result['key_sentences']),
                result['final_summary']
            ))
        
        # Save summary file for this assignment if provided
        summary_filename = None
        student_summary_text = None
        if summary_files and idx < len(summary_files):
            summary_file = summary_files[idx]
            if summary_file and summary_file.filename:
                summary_filename = secure_filename(summary_file.filename)
                summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_filename)
                summary_file.save(summary_path)
                # Extract text from the summary file
                if summary_path.endswith('.docx'):
                    student_summary_text = read_docx(summary_path)
                elif summary_path.endswith('.pdf'):
                    student_summary_text = read_pdf(summary_path)
        
        # Save summary using the best k value result
        best_result = processing_results[best_k]
        conn.execute('INSERT INTO Summary (assignment_id, summary_text, created_at, summary_file) VALUES (?, ?, datetime("now"), ?)',
                     (assignment_id, best_result['final_summary'], summary_filename))
        
        # Save student summary to evaluation table if provided
        if student_summary_text:
            # Calculate ROUGE scores
            rouge_scores = compute_rouge(answer_key, student_summary_text)
            
            conn.execute('''
                INSERT INTO Evaluation (
                    assignment_id, student_summary, summary_file,
                    rouge1_precision, rouge1_recall, rouge1_f1,
                    rouge2_precision, rouge2_recall, rouge2_f1,
                    rougeL_precision, rougeL_recall, rougeL_f1,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))
            ''', (
                assignment_id, student_summary_text, summary_filename,
                rouge_scores['rouge1'].precision, rouge_scores['rouge1'].recall, rouge_scores['rouge1'].fmeasure,
                rouge_scores['rouge2'].precision, rouge_scores['rouge2'].recall, rouge_scores['rouge2'].fmeasure,
                rouge_scores['rougeL'].precision, rouge_scores['rougeL'].recall, rouge_scores['rougeL'].fmeasure
            ))
    
    # Save answer key (once per batch)
    conn.execute('INSERT INTO Answer_Key (lecturer_id, text_key, uploaded_at) VALUES (?, ?, datetime("now"))',
                 (lecturer_id, answer_key))
    conn.commit()
    conn.close()

    flash('Batch assignments uploaded and processed successfully!')
    return redirect(url_for('upload'))

# Helper for grading (cosine similarity, etc.)
def compute_grading(summary, answer_key):
    def get_embedding(text):
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    emb_summary = get_embedding(summary)
    emb_key = get_embedding(answer_key)
    sim = cosine_similarity([emb_summary], [emb_key])[0][0]
    percent_sim = sim * 100
    if percent_sim <= 20:
        grade, value = "Poor", 1
    elif percent_sim <= 40:
        grade, value = "Bad", 2
    elif percent_sim <= 60:
        grade, value = "Fair", 3
    elif percent_sim <= 80:
        grade, value = "Good", 4
    else:
        grade, value = "Excellent", 5
    return percent_sim, grade, value

if __name__ == '__main__':
    app.run(debug=True)
