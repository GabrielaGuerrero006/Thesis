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

def get_num_mangos_procesados(lote_number):
    """
    Retorna el número de mangos procesados (IDs únicos) para un lote dado.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de item_id únicos para ese lote
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ?', (int(lote_number),))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_num_detecciones_lote(lote_number):
    """
    Retorna el número de detecciones totales para un lote dado.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros para ese lote
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ?', (int(lote_number),))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_num_exportables_no_exportables(lote_number):
    """
    Retorna el número de detecciones de mangos exportables y no exportables para un lote dado.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        dict: {'exportable': int, 'no_exportable': int}
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ? AND detection_type = ?', (int(lote_number), 'exportable'))
    exportable = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ? AND detection_type = ?', (int(lote_number), 'no_exportable'))
    no_exportable = cursor.fetchone()[0]
    conn.close()
    return {'exportable': exportable, 'no_exportable': no_exportable}

def get_num_verdes_maduros(lote_number):
    """
    Retorna el número de detecciones de mangos verdes y maduros para un lote dado.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        dict: {'mango_verde': int, 'mango_maduro': int}
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ? AND detection_type = ?', (int(lote_number), 'mango_verde'))
    mango_verde = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ? AND detection_type = ?', (int(lote_number), 'mango_maduro'))
    mango_maduro = cursor.fetchone()[0]
    conn.close()
    return {'mango_verde': mango_verde, 'mango_maduro': mango_maduro}

def get_num_con_defectos_sin_defectos(lote_number):
    """
    Retorna el número de detecciones de mangos con y sin defectos para un lote dado.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        dict: {'mango_con_defectos': int, 'mango_sin_defectos': int}
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ? AND detection_type = ?', (int(lote_number), 'mango_con_defectos'))
    mango_con_defectos = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM detections WHERE lote_number = ? AND detection_type = ?', (int(lote_number), 'mango_sin_defectos'))
    mango_sin_defectos = cursor.fetchone()[0]
    conn.close()
    return {'mango_con_defectos': mango_con_defectos, 'mango_sin_defectos': mango_sin_defectos}

def get_fecha_procesado_lote(lote_number):
    """
    Retorna la fecha de procesado del lote (primer registro encontrado).
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        str: fecha (YYYY-MM-DD) o None si no hay registros
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT detection_date FROM detections WHERE lote_number = ? ORDER BY detection_date ASC LIMIT 1', (int(lote_number),))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_confianza_promedio_lote(lote_number):
    """
    Retorna el porcentaje de confianza promedio de todos los modelos para un lote dado, ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND confidence > 0', (int(lote_number),))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0

def get_confianza_promedio_exportabilidad(lote_number):
    """
    Retorna el porcentaje de confianza promedio del modelo exportabilidad para un lote dado, ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND model_name = ? AND confidence > 0', (int(lote_number), 'exportabilidad.pt'))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0

def get_confianza_promedio_madurez(lote_number):
    """
    Retorna el porcentaje de confianza promedio del modelo madurez para un lote dado, ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND model_name = ? AND confidence > 0', (int(lote_number), 'madurez.pt'))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0

def get_confianza_promedio_defectos(lote_number):
    """
    Retorna el porcentaje de confianza promedio del modelo defectos para un lote dado, ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND model_name = ? AND confidence > 0', (int(lote_number), 'defectos.pt'))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0

def get_cantidad_mangos_exportables_lote(lote_number):
    """
    Retorna la cantidad de mangos exportables del lote usando el modelo exportabilidad.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros con detection_type = 'exportable' y model_name = 'exportabilidad.pt'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'exportabilidad.pt', 'exportable'))
    cantidad = cursor.fetchone()[0]
    conn.close()
    return cantidad

def get_cantidad_mangos_no_exportables_lote(lote_number):
    """
    Retorna la cantidad de mangos no exportables del lote usando el modelo exportabilidad.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros con detection_type = 'no_exportable' y model_name = 'exportabilidad.pt'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'exportabilidad.pt', 'no_exportable'))
    cantidad = cursor.fetchone()[0]
    conn.close()
    return cantidad

def get_cantidad_mangos_maduros_lote(lote_number):
    """
    Retorna la cantidad de mangos maduros del lote usando el modelo madurez.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros con detection_type = 'mango_maduro' y model_name = 'madurez.pt'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'madurez.pt', 'mango_maduro'))
    cantidad = cursor.fetchone()[0]
    conn.close()
    return cantidad

def get_cantidad_mangos_verdes_lote(lote_number):
    """
    Retorna la cantidad de mangos verdes del lote usando el modelo madurez.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros con detection_type = 'mango_verde' y model_name = 'madurez.pt'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'madurez.pt', 'mango_verde'))
    cantidad = cursor.fetchone()[0]
    conn.close()
    return cantidad

def get_cantidad_mangos_con_defecto_lote(lote_number):
    """
    Retorna la cantidad de mangos con defecto del lote usando el modelo defectos.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros con detection_type = 'mango_con_defecto' y model_name = 'defectos.pt'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_con_defecto'))
    cantidad = cursor.fetchone()[0]
    conn.close()
    return cantidad

def get_cantidad_mangos_sin_defecto_lote(lote_number):
    """
    Retorna la cantidad de mangos sin defecto del lote usando el modelo defectos.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        int: cantidad de registros con detection_type = 'mango_sin_defecto' y model_name = 'defectos.pt'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_sin_defecto'))
    cantidad = cursor.fetchone()[0]
    conn.close()
    return cantidad

def get_porcentaje_mangos_exportables_lote(lote_number):
    """
    Retorna el porcentaje de mangos exportables del lote usando el modelo exportabilidad.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'exportabilidad.pt', 'exportable', 'no_exportable'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'exportabilidad.pt', 'exportable'))
    exportable = cursor.fetchone()[0]
    conn.close()
    return round((exportable / total) * 100, 2) if total > 0 else 0.0

def get_porcentaje_mangos_no_exportables_lote(lote_number):
    """
    Retorna el porcentaje de mangos no exportables del lote usando el modelo exportabilidad.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'exportabilidad.pt', 'exportable', 'no_exportable'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'exportabilidad.pt', 'no_exportable'))
    no_exportable = cursor.fetchone()[0]
    conn.close()
    return round((no_exportable / total) * 100, 2) if total > 0 else 0.0

def get_porcentaje_mangos_verdes_lote(lote_number):
    """
    Retorna el porcentaje de mangos verdes del lote usando el modelo madurez.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'madurez.pt', 'mango_verde', 'mango_maduro'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'madurez.pt', 'mango_verde'))
    mango_verde = cursor.fetchone()[0]
    conn.close()
    return round((mango_verde / total) * 100, 2) if total > 0 else 0.0

def get_porcentaje_mangos_maduros_lote(lote_number):
    """
    Retorna el porcentaje de mangos maduros del lote usando el modelo madurez.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'madurez.pt', 'mango_verde', 'mango_maduro'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'madurez.pt', 'mango_maduro'))
    mango_maduro = cursor.fetchone()[0]
    conn.close()
    return round((mango_maduro / total) * 100, 2) if total > 0 else 0.0

def get_porcentaje_mangos_con_defecto_lote(lote_number):
    """
    Retorna el porcentaje de mangos con defecto del lote usando el modelo defectos.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'defectos.pt', 'mango_con_defecto', 'mango_sin_defecto'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_con_defecto'))
    mango_con_defecto = cursor.fetchone()[0]
    conn.close()
    return round((mango_con_defecto / total) * 100, 2) if total > 0 else 0.0

def get_porcentaje_mangos_sin_defecto_lote(lote_number):
    """
    Retorna el porcentaje de mangos sin defecto del lote usando el modelo defectos.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'defectos.pt', 'mango_con_defecto', 'mango_sin_defecto'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_sin_defecto'))
    mango_sin_defecto = cursor.fetchone()[0]
    conn.close()
    return round((mango_sin_defecto / total) * 100, 2) if total > 0 else 0.0

def get_ids_lote(lote_number):
    """
    Retorna una lista de todos los item_id diferentes para un lote dado.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
    Returns:
        list: lista de item_id únicos para ese lote
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT item_id FROM detections WHERE lote_number = ? ORDER BY item_id', (int(lote_number),))
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids
