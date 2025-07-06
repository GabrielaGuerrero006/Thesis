import sqlite3
import os
import pandas as pd
from datetime import datetime

#Ruta de la BD
def get_db_path():
    """Retorna la ruta al archivo de la base de datos"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detections.db')

#Inicialización de la BD
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
    
    # Crear nueva tabla para imágenes capturadas (ahora con columna BLOB para la imagen)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captured_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lote_number INTEGER,
            item_id INTEGER,
            capture_date DATE,
            capture_time TIME,
            image_path TEXT UNIQUE,
            image_blob BLOB
        )
    ''')
    
    conn.commit()
    conn.close()

#Guardado de las detecciones
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

def save_image_db(lote_number, item_id, image_path):
    """
    Guarda la información de una imagen capturada en la base de datos.
    
    Args:
        lote_number (int): Número de lote.
        item_id (int): ID del mango.
        image_path (str): Ruta relativa o absoluta del archivo de imagen.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    current_time = datetime.now()
    capture_date = current_time.strftime('%Y-%m-%d')
    capture_time = current_time.strftime('%H:%M:%S')
    
    # Mensaje de depuración para mostrar los datos que se intentan guardar
    print(f"DEBUG: Intentando guardar imagen en DB: Lote={lote_number}, ID={item_id}, Fecha={capture_date}, Hora={capture_time}, Ruta={image_path}")
    
    try:
        # Leer la imagen como binario (BLOB)
        with open(image_path, 'rb') as f:
            image_blob = f.read()
        cursor.execute(
            'INSERT INTO captured_images (lote_number, item_id, capture_date, capture_time, image_path, image_blob) VALUES (?, ?, ?, ?, ?, ?)',
            (lote_number, item_id, capture_date, capture_time, image_path, image_blob)
        )
        conn.commit()
        print(f"DEBUG: Imagen {image_path} guardada exitosamente en la base de datos como BLOB.")
    except sqlite3.IntegrityError as e:
        # Esto se activaría si image_path ya existe debido a la restricción UNIQUE
        print(f"ERROR: sqlite3.IntegrityError al guardar imagen {image_path}: {e}. La imagen ya existe o hay un conflicto de clave única.")
    except Exception as e:
        # Captura cualquier otra excepción durante el proceso de guardado
        print(f"ERROR: Error general al guardar la imagen en la base de datos: {e}")
    finally:
        conn.close()

def get_lotes():
    """Obtiene todos los números de lote únicos de la base de datos"""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute('SELECT DISTINCT lote_number FROM detections ORDER BY lote_number')
    lotes = [str(row[0]) for row in cursor.fetchall()]
    
    conn.close()
    return lotes

#Funciones para lotes
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
        dict: _defectos': int, 'mango_sin_defectos': int}
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
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_con_defectos'))
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
    cursor.execute('''SELECT COUNT(DISTINCT item_id) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_sin_defectos'))
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
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'defectos.pt', 'mango_con_defectos', 'mango_sin_defectos'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_con_defectos'))
    mango_con_defecto = cursor.fetchone()[0]
    conn.close()
    return round((mango_con_defecto / total) * 100, 2) if total > 0 else 0.0

def get_porcentaje_mangos_sin_defecto_lote(lote_number):
    """
    Retorna el porcentaje de mangos sin defecto del lote usando el modelo defectos.pt.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND (detection_type = ? OR detection_type = ?)''', (int(lote_number), 'defectos.pt', 'mango_con_defectos', 'mango_sin_defectos'))
    total = cursor.fetchone()[0]
    cursor.execute('''SELECT COUNT(*) FROM detections WHERE lote_number = ? AND model_name = ? AND detection_type = ?''', (int(lote_number), 'defectos.pt', 'mango_sin_defectos'))
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

#Funciones para datos de forma unitaria
def get_fecha_deteccion_lote_id(lote_number, item_id):
    """
    Retorna la fecha más antigua de detección para un lote y item_id dados.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
        item_id (str o int): ID seleccionado por el usuario
    Returns:
        str: fecha (YYYY-MM-DD) más antigua o None si no hay registros
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT detection_date FROM detections WHERE lote_number = ? AND item_id = ? ORDER BY detection_date ASC LIMIT 1', (int(lote_number), int(item_id)))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_exportabilidad_mango(lote_number, item_id):
    """
    Determina la exportabilidad del mango para un lote y item_id dados según el modelo exportabilidad.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
        item_id (str o int): ID seleccionado por el usuario
    Returns:
        str: 'Exportable', 'No Exportable', 'Nulo' o 'Sin datos suficientes'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT detection_type FROM detections WHERE lote_number = ? AND item_id = ? AND model_name = ?''', (int(lote_number), int(item_id), 'exportabilidad.pt'))
    resultados = [row[0] for row in cursor.fetchall()]
    conn.close()
    if not resultados:
        return 'Sin datos suficientes'
    total = len(resultados)
    exportable = sum(1 for r in resultados if r == 'exportable')
    no_exportable = sum(1 for r in resultados if r == 'no_exportable')
    nulo = sum(1 for r in resultados if r == 'no detections')
    if exportable / total >= 0.8:
        return 'Exportable'
    elif no_exportable / total >= 0.8:
        return 'No Exportable'
    elif nulo / total >= 0.8:
        return 'Nulo'
    else:
        return 'Sin datos suficientes'

def get_madurez_mango(lote_number, item_id):
    """
    Determina la madurez del mango para un lote y item_id dados según el modelo madurez.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
        item_id (str o int): ID seleccionado por el usuario
    Returns:
        str: 'Verde', 'Maduro', 'Nulo' o 'Sin datos suficientes'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT detection_type FROM detections WHERE lote_number = ? AND item_id = ? AND model_name = ?''', (int(lote_number), int(item_id), 'madurez.pt'))
    resultados = [row[0] for row in cursor.fetchall()]
    conn.close()
    if not resultados:
        return 'Sin datos suficientes'
    total = len(resultados)
    verde = sum(1 for r in resultados if r == 'mango_verde')
    maduro = sum(1 for r in resultados if r == 'mango_maduro')
    nulo = sum(1 for r in resultados if r == 'no detections')
    if verde / total >= 0.8:
        return 'Verde'
    elif maduro / total >= 0.8:
        return 'Maduro'
    elif nulo / total >= 0.8:
        return 'Nulo'
    else:
        return 'Sin datos suficientes'

def get_defectos_mango(lote_number, item_id):
    """
    Determina si el mango tiene defectos para un lote y item_id dados según el modelo defectos.pt.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario
        item_id (str o int): ID seleccionado por el usuario
    Returns:
        str: 'No', 'Si', 'Nulo' o 'Sin datos suficientes'
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT detection_type FROM detections WHERE lote_number = ? AND item_id = ? AND model_name = ?''', (int(lote_number), int(item_id), 'defectos.pt'))
    resultados = [row[0] for row in cursor.fetchall()]
    conn.close()
    if not resultados:
        return 'Sin datos suficientes'
    total = len(resultados)
    sin_defectos = sum(1 for r in resultados if r == 'mango_sin_defecto')
    con_defectos = sum(1 for r in resultados if r == 'mango_con_defectos')
    nulo = sum(1 for r in resultados if r == 'no detections')
    if sin_defectos / total >= 0.8:
        return 'No'
    elif con_defectos / total >= 0.8:
        return 'Si'
    elif nulo / total >= 0.8:
        return 'Nulo'
    else:
        return 'Sin datos suficientes'

def get_confianza_promedio_exportabilidad_mango(lote_number, item_id):
    """
    Retorna el porcentaje de confianza promedio del modelo exportabilidad para un mango específico (lote + item_id), ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote
        item_id (str o int): ID del mango
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND item_id = ? AND model_name = ? AND confidence > 0', (int(lote_number), int(item_id), 'exportabilidad.pt'))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0

def get_confianza_promedio_madurez_mango(lote_number, item_id):
    """
    Retorna el porcentaje de confianza promedio del modelo madurez para un mango específico (lote + item_id), ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote
        item_id (str o int): ID del mango
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND item_id = ? AND model_name = ? AND confidence > 0', (int(lote_number), int(item_id), 'madurez.pt'))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0

def get_confianza_promedio_defectos_mango(lote_number, item_id):
    """
    Retorna el porcentaje de confianza promedio del modelo defectos para un mango específico (lote + item_id), ignorando valores de confianza igual a 0.
    Args:
        lote_number (str o int): Número de lote
        item_id (str o int): ID del mango
    Returns:
        float: porcentaje de confianza promedio (ej: 93.37)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM detections WHERE lote_number = ? AND item_id = ? AND model_name = ? AND confidence > 0', (int(lote_number), int(item_id), 'defectos.pt'))
    confidences = [row[0] for row in cursor.fetchall()]
    conn.close()
    if confidences:
        avg = sum(confidences) / len(confidences)
        return round(avg * 100, 2)
    return 0.0