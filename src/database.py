import sqlite3
import os
from datetime import datetime

def get_db_connection(db_path="database.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path="database.db"):
    """
    Creates the logs table if it does not already exist.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            doctor_notes TEXT,
            image_path TEXT,
            cam_path TEXT,
            mask_path TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"[Database] SQLite initialized successfully at '{db_path}'")

def save_record(patient_name, age, gender, prediction, confidence, model_used, doctor_notes, image_path, cam_path, mask_path, db_path="database.db"):
    """
    Saves a diagnostic prediction record to the database.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO prediction_logs (
            patient_name, age, gender, prediction, confidence, model_used, timestamp, doctor_notes, image_path, cam_path, mask_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_name, age, gender, prediction, confidence, model_used, timestamp, doctor_notes, image_path, cam_path, mask_path))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_records(db_path="database.db"):
    """
    Retrieves all logs from the database, ordered by timestamp descending.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_logs ORDER BY timestamp DESC")
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def search_records(query_str=None, disease=None, model=None, db_path="database.db"):
    """
    Searches patient records with dynamic filters.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM prediction_logs WHERE 1=1"
    params = []
    
    if query_str:
        query += " AND (patient_name LIKE ? OR doctor_notes LIKE ?)"
        params.append(f"%{query_str}%")
        params.append(f"%{query_str}%")
        
    if disease:
        query += " AND prediction = ?"
        params.append(disease)
        
    if model:
        query += " AND model_used = ?"
        params.append(model)
        
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

def delete_record(record_id, db_path="database.db"):
    """
    Deletes a record by its unique ID.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_logs WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return True
