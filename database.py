import sqlite3
import os
import pandas as pd
from datetime import datetime

def get_db_path():
    """Retorna la ruta al archivo de la base de datos"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detections.db')

def init_db():
    """Inicializa la base de datos y crea las tablas necesarias si no existen"""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Crear tabla de detecciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lote_number INTEGER,
            item_id INTEGER,
            detection_date DATE,
            detection_time TIME,
            model_name TEXT,
            detection_type TEXT,
            confidence REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_detections_db(detections_list):
    """
    Guarda una lista de detecciones en la base de datos
    
    Args:
        detections_list: Lista de detecciones en formato
        [lote, id, fecha, hora, modelo, tipo_deteccion, confianza]
    """
    if not detections_list:
        return
        
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.executemany(
        'INSERT INTO detections (lote_number, item_id, detection_date, detection_time, model_name, detection_type, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)',
        detections_list
    )
    
    conn.commit()
    conn.close()

def get_lotes():
    """Obtiene todos los números de lote únicos de la base de datos"""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute('SELECT DISTINCT lote_number FROM detections ORDER BY lote_number')
    lotes = [str(row[0]) for row in cursor.fetchall()]
    
    conn.close()
    return lotes

def get_lote_data(lote_number):
    """
    Obtiene todos los datos de un lote específico y los retorna como DataFrame
    
    Args:
        lote_number (str): Número del lote a obtener
    
    Returns:
        pandas.DataFrame: DataFrame con los datos del lote
    """
    conn = sqlite3.connect(get_db_path())
    
    query = '''
    SELECT lote_number as Lote, 
           item_id as ID, 
           detection_date as Fecha, 
           detection_time as Hora, 
           model_name as Modelo, 
           detection_type as Tipo_Deteccion, 
           confidence as Confianza
    FROM detections 
    WHERE lote_number = ?
    '''
    
    df = pd.read_sql_query(query, conn, params=(int(lote_number),))
    conn.close()
    
    return df

def get_ids_for_lote(lote_number):
    """
    Obtiene todos los IDs únicos para un lote específico
    
    Args:
        lote_number (str): Número del lote
    
    Returns:
        tuple: (cantidad_ids, lista_ids)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT DISTINCT item_id FROM detections WHERE lote_number = ?',
        (int(lote_number),)
    )
    
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return len(ids), ids
