import sqlite3
from datetime import datetime

conn = sqlite3.connect('projek_skripsi.db')
c = conn.cursor()

# Lecturer table
c.execute('''
CREATE TABLE IF NOT EXISTS Lecturer (
    lecturer_id INTEGER PRIMARY KEY NOT NULL,
    lecturer_name VARCHAR(50) NOT NULL,
    lecturer_email VARCHAR(50) NOT NULL UNIQUE,
    lecturer_password VARCHAR(50) NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
''')

# Activity_Log table
c.execute('''
CREATE TABLE IF NOT EXISTS Activity_Log (
    log_id INTEGER PRIMARY KEY NOT NULL,
    lecturer_id INTEGER NOT NULL,
    activity VARCHAR(50) NOT NULL,
    created_at TIMESTAMP,
    FOREIGN KEY (lecturer_id) REFERENCES Lecturer(lecturer_id)
)
''')

# Student_Assignment table
c.execute('''
CREATE TABLE IF NOT EXISTS Student_Assignment (
    assignment_id INTEGER PRIMARY KEY NOT NULL,
    lecturer_id INTEGER NOT NULL,
    student_name VARCHAR(50) NOT NULL,
    assignment TEXT NOT NULL,
    uploaded_at TIMESTAMP,
    FOREIGN KEY (lecturer_id) REFERENCES Lecturer(lecturer_id)
)
''')

# Summary table
c.execute('''
CREATE TABLE IF NOT EXISTS Summary (
    summary_id INTEGER PRIMARY KEY NOT NULL,
    assignment_id INTEGER NOT NULL,
    summary_text TEXT NOT NULL,
    summary_file TEXT NOT NULL,
    created_at TIMESTAMP,
    FOREIGN KEY (assignment_id) REFERENCES Student_Assignment(assignment_id)
)
''')

# Answer_Key table
c.execute('''
CREATE TABLE IF NOT EXISTS Answer_Key (
    key_id INTEGER PRIMARY KEY NOT NULL,
    lecturer_id INTEGER NOT NULL,
    text_key TEXT NOT NULL,
    uploaded_at TIMESTAMP,
    FOREIGN KEY (lecturer_id) REFERENCES Lecturer(lecturer_id)
)
''')

# Similarity table
c.execute('''
CREATE TABLE IF NOT EXISTS Similarity (
    similarity_id INTEGER PRIMARY KEY NOT NULL,
    summary_id INTEGER NOT NULL,
    key_id INTEGER NOT NULL,
    score FLOAT NOT NULL,
    checked_at TIMESTAMP,
    FOREIGN KEY (summary_id) REFERENCES Summary(summary_id),
    FOREIGN KEY (key_id) REFERENCES Answer_Key(key_id)
)
''')

# Evaluation table
c.execute('''
CREATE TABLE IF NOT EXISTS Evaluation (
    evaluation_id INTEGER PRIMARY KEY NOT NULL,
    assignment_id INTEGER NOT NULL,
    student_summary TEXT NOT NULL,
    summary_file TEXT NOT NULL,
    rouge1_precision FLOAT,
    rouge1_recall FLOAT,
    rouge1_f1 FLOAT,
    rouge2_precision FLOAT,
    rouge2_recall FLOAT,
    rouge2_f1 FLOAT,
    rougeL_precision FLOAT,
    rougeL_recall FLOAT,
    rougeL_f1 FLOAT,
    created_at TIMESTAMP,
    FOREIGN KEY (assignment_id) REFERENCES Student_Assignment(assignment_id)
)
''')

# Processing_Steps table
c.execute('''
CREATE TABLE IF NOT EXISTS Processing_Steps (
    process_id INTEGER PRIMARY KEY NOT NULL,
    assignment_id INTEGER NOT NULL,
    cleaned_text TEXT NOT NULL,
    extracted_sentences TEXT NOT NULL,
    embeddings TEXT NOT NULL,
    silhouette_score FLOAT,
    num_clusters INTEGER NOT NULL,
    cluster_labels TEXT NOT NULL,
    key_sentences TEXT NOT NULL,
    final_summary TEXT NOT NULL,
    created_at TIMESTAMP,
    FOREIGN KEY (assignment_id) REFERENCES Student_Assignment(assignment_id)
)
''')

# Insert default lecturer for testing if not exists
c.execute('''
INSERT OR IGNORE INTO Lecturer (lecturer_id, lecturer_name, lecturer_email, lecturer_password, created_at, updated_at)
VALUES (1, 'Admin', 'admin@example.com', 'admin123', ?, ?)
''', (datetime.now(), datetime.now()))

conn.commit()
conn.close() 