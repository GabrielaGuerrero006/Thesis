import base64
import sqlite3
import os
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

def get_images_by_lote_and_id(lote_number, item_id):
    """
    Obtiene las imágenes (BLOB) asociadas a un lote y un ID específico.
    Devuelve una lista de imágenes codificadas en base64 (para mostrar en HTML).
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('''SELECT image_blob FROM captured_images WHERE lote_number = ? AND item_id = ? ORDER BY id ASC''', (int(lote_number), int(item_id)))
    blobs = [row[0] for row in cursor.fetchall()]
    conn.close()
    # Convertir cada blob a base64 para mostrar en HTML
    images_base64 = []
    for blob in blobs:
        if blob:
            images_base64.append(base64.b64encode(blob).decode('utf-8'))
    return images_base64

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
    Retorna el número de detecciones de mangos exportables y no exportables para un lote dado,
    aplicando un filtro de "mayoría de votos" por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        dict: {'exportable': int, 'no_exportable': int}
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'exportable' THEN 1 ELSE 0 END) AS exportable_count,
                SUM(CASE WHEN detection_type = 'no_exportable' THEN 1 ELSE 0 END) AS no_exportable_count
            FROM
                detections
            WHERE
                lote_number = ?
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN T2.detection_type = 'exportable' AND T1.exportable_count > T1.no_exportable_count THEN 1 ELSE 0 END) AS exportables,
            SUM(CASE WHEN T2.detection_type = 'no_exportable' AND T1.no_exportable_count > T1.exportable_count THEN 1 ELSE 0 END) AS no_exportables
        FROM
            item_votes AS T1
        JOIN
            detections AS T2 ON T1.item_id = T2.item_id
        WHERE
            T2.lote_number = ?;
        """
        
        cursor.execute(query, (int(lote_number), int(lote_number)))
        result = cursor.fetchone()

        exportable = result[0] if result[0] is not None else 0
        no_exportable = result[1] if result[1] is not None else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        exportable = 0
        no_exportable = 0
    finally:
        conn.close()

    return {'exportable': exportable, 'no_exportable': no_exportable}

def get_num_verdes_maduros(lote_number):
    """
    Retorna el número de detecciones de mangos verdes y maduros para un lote dado,
    aplicando un filtro de "mayoría de votos" por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        dict: {'mango_verde': int, 'mango_maduro': int}
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_verde' THEN 1 ELSE 0 END) AS verde_count,
                SUM(CASE WHEN detection_type = 'mango_maduro' THEN 1 ELSE 0 END) AS maduro_count
            FROM
                detections
            WHERE
                lote_number = ?
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN T2.detection_type = 'mango_verde' AND T1.verde_count > T1.maduro_count THEN 1 ELSE 0 END) AS verdes,
            SUM(CASE WHEN T2.detection_type = 'mango_maduro' AND T1.maduro_count > T1.verde_count THEN 1 ELSE 0 END) AS maduros
        FROM
            item_votes AS T1
        JOIN
            detections AS T2 ON T1.item_id = T2.item_id
        WHERE
            T2.lote_number = ?;
        """
        
        cursor.execute(query, (int(lote_number), int(lote_number)))
        result = cursor.fetchone()

        mango_verde = result[0] if result[0] is not None else 0
        mango_maduro = result[1] if result[1] is not None else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        mango_verde = 0
        mango_maduro = 0
    finally:
        conn.close()

    return {'mango_verde': mango_verde, 'mango_maduro': mango_maduro}

def get_num_con_defectos_sin_defectos(lote_number):
    """
    Retorna el número de detecciones de mangos con y sin defectos para un lote dado,
    aplicando un filtro de "mayoría de votos" por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        dict: {'mango_con_defectos': int, 'mango_sin_defectos': int}
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_con_defectos' THEN 1 ELSE 0 END) AS con_defectos_count,
                SUM(CASE WHEN detection_type = 'mango_sin_defectos' THEN 1 ELSE 0 END) AS sin_defectos_count
            FROM
                detections
            WHERE
                lote_number = ?
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN T2.detection_type = 'mango_con_defectos' AND T1.con_defectos_count > T1.sin_defectos_count THEN 1 ELSE 0 END) AS con_defectos,
            SUM(CASE WHEN T2.detection_type = 'mango_sin_defectos' AND T1.sin_defectos_count > T1.con_defectos_count THEN 1 ELSE 0 END) AS sin_defectos
        FROM
            item_votes AS T1
        JOIN
            detections AS T2 ON T1.item_id = T2.item_id
        WHERE
            T2.lote_number = ?;
        """
        
        cursor.execute(query, (int(lote_number), int(lote_number)))
        result = cursor.fetchone()

        mango_con_defectos = result[0] if result[0] is not None else 0
        mango_sin_defectos = result[1] if result[1] is not None else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        mango_con_defectos = 0
        mango_sin_defectos = 0
    finally:
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
    Retorna la cantidad de mangos únicos clasificados como exportables
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        int: Cantidad de mangos únicos de tipo 'exportable'.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'exportable' THEN 1 ELSE 0 END) AS exportable_count,
                SUM(CASE WHEN detection_type = 'no_exportable' THEN 1 ELSE 0 END) AS no_exportable_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'exportabilidad.pt'
            GROUP BY
                item_id
        )
        SELECT
            COUNT(item_id)
        FROM
            item_votes
        WHERE
            exportable_count > no_exportable_count;
        """
        
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()
        cantidad = result[0] if result else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        cantidad = 0
    finally:
        conn.close()

    return cantidad

def get_cantidad_mangos_no_exportables_lote(lote_number):
    """
    Retorna la cantidad de mangos únicos clasificados como no exportables
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        int: Cantidad de mangos únicos de tipo 'no_exportable'.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'exportable' THEN 1 ELSE 0 END) AS exportable_count,
                SUM(CASE WHEN detection_type = 'no_exportable' THEN 1 ELSE 0 END) AS no_exportable_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'exportabilidad.pt'
            GROUP BY
                item_id
        )
        SELECT
            COUNT(item_id)
        FROM
            item_votes
        WHERE
            no_exportable_count > exportable_count;
        """
        
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()
        cantidad = result[0] if result else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        cantidad = 0
    finally:
        conn.close()

    return cantidad

def get_cantidad_mangos_maduros_lote(lote_number):
    """
    Retorna la cantidad de mangos únicos clasificados como maduros
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        int: Cantidad de mangos únicos de tipo 'mango_maduro'.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_maduro' THEN 1 ELSE 0 END) AS maduro_count,
                SUM(CASE WHEN detection_type = 'mango_verde' THEN 1 ELSE 0 END) AS verde_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'madurez.pt'
            GROUP BY
                item_id
        )
        SELECT
            COUNT(item_id)
        FROM
            item_votes
        WHERE
            maduro_count > verde_count;
        """
        
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()
        cantidad = result[0] if result else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        cantidad = 0
    finally:
        conn.close()

    return cantidad

def get_cantidad_mangos_verdes_lote(lote_number):
    """
    Retorna la cantidad de mangos únicos clasificados como verdes
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        int: Cantidad de mangos únicos de tipo 'mango_verde'.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_verde' THEN 1 ELSE 0 END) AS verde_count,
                SUM(CASE WHEN detection_type = 'mango_maduro' THEN 1 ELSE 0 END) AS maduro_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'madurez.pt'
            GROUP BY
                item_id
        )
        SELECT
            COUNT(item_id)
        FROM
            item_votes
        WHERE
            verde_count > maduro_count;
        """
        
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()
        cantidad = result[0] if result else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        cantidad = 0
    finally:
        conn.close()

    return cantidad

def get_cantidad_mangos_con_defecto_lote(lote_number):
    """
    Retorna la cantidad de mangos únicos clasificados con defecto
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        int: Cantidad de mangos únicos de tipo 'mango_con_defectos'.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_con_defectos' THEN 1 ELSE 0 END) AS con_defectos_count,
                SUM(CASE WHEN detection_type = 'mango_sin_defectos' THEN 1 ELSE 0 END) AS sin_defectos_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'defectos.pt'
            GROUP BY
                item_id
        )
        SELECT
            COUNT(item_id)
        FROM
            item_votes
        WHERE
            con_defectos_count > sin_defectos_count;
        """
        
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()
        cantidad = result[0] if result else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        cantidad = 0
    finally:
        conn.close()

    return cantidad

def get_cantidad_mangos_sin_defecto_lote(lote_number):
    """
    Retorna la cantidad de mangos únicos clasificados sin defecto
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    Args:
        lote_number (str o int): Número de lote seleccionado por el usuario.
    Returns:
        int: Cantidad de mangos únicos de tipo 'mango_sin_defectos'.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_sin_defectos' THEN 1 ELSE 0 END) AS sin_defectos_count,
                SUM(CASE WHEN detection_type = 'mango_con_defectos' THEN 1 ELSE 0 END) AS con_defectos_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'defectos.pt'
            GROUP BY
                item_id
        )
        SELECT
            COUNT(item_id)
        FROM
            item_votes
        WHERE
            sin_defectos_count > con_defectos_count;
        """
        
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()
        cantidad = result[0] if result else 0
        
    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        cantidad = 0
    finally:
        conn.close()

    return cantidad

def get_porcentaje_mangos_exportables_lote(lote_number):
    """
    Retorna el porcentaje de mangos únicos clasificados como exportables
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'exportable' THEN 1 ELSE 0 END) AS exportable_count,
                SUM(CASE WHEN detection_type = 'no_exportable' THEN 1 ELSE 0 END) AS no_exportable_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'exportabilidad.pt'
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN exportable_count > no_exportable_count THEN 1 ELSE 0 END) AS exportable_mangos,
            SUM(CASE WHEN no_exportable_count > exportable_count THEN 1 ELSE 0 END) AS no_exportable_mangos
        FROM
            item_votes;
        """
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()

        exportable_mangos = result[0] if result and result[0] is not None else 0
        no_exportable_mangos = result[1] if result and result[1] is not None else 0
        total_mangos = exportable_mangos + no_exportable_mangos
        
        return round((exportable_mangos / total_mangos) * 100, 2) if total_mangos > 0 else 0.0

    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        return 0.0
    finally:
        conn.close()

def get_porcentaje_mangos_no_exportables_lote(lote_number):
    """
    Retorna el porcentaje de mangos únicos clasificados como no exportables
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'exportable' THEN 1 ELSE 0 END) AS exportable_count,
                SUM(CASE WHEN detection_type = 'no_exportable' THEN 1 ELSE 0 END) AS no_exportable_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'exportabilidad.pt'
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN exportable_count > no_exportable_count THEN 1 ELSE 0 END) AS exportable_mangos,
            SUM(CASE WHEN no_exportable_count > exportable_count THEN 1 ELSE 0 END) AS no_exportable_mangos
        FROM
            item_votes;
        """
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()

        exportable_mangos = result[0] if result and result[0] is not None else 0
        no_exportable_mangos = result[1] if result and result[1] is not None else 0
        total_mangos = exportable_mangos + no_exportable_mangos
        
        return round((no_exportable_mangos / total_mangos) * 100, 2) if total_mangos > 0 else 0.0

    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        return 0.0
    finally:
        conn.close()

def get_porcentaje_mangos_verdes_lote(lote_number):
    """
    Retorna el porcentaje de mangos únicos clasificados como verdes
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_verde' THEN 1 ELSE 0 END) AS verde_count,
                SUM(CASE WHEN detection_type = 'mango_maduro' THEN 1 ELSE 0 END) AS maduro_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'madurez.pt'
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN verde_count > maduro_count THEN 1 ELSE 0 END) AS mangos_verdes,
            SUM(CASE WHEN maduro_count > verde_count THEN 1 ELSE 0 END) AS mangos_maduros
        FROM
            item_votes;
        """
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()

        mangos_verdes = result[0] if result and result[0] is not None else 0
        mangos_maduros = result[1] if result and result[1] is not None else 0
        total_mangos = mangos_verdes + mangos_maduros
        
        return round((mangos_verdes / total_mangos) * 100, 2) if total_mangos > 0 else 0.0

    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        return 0.0
    finally:
        conn.close()

def get_porcentaje_mangos_maduros_lote(lote_number):
    """
    Retorna el porcentaje de mangos únicos clasificados como maduros
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_verde' THEN 1 ELSE 0 END) AS verde_count,
                SUM(CASE WHEN detection_type = 'mango_maduro' THEN 1 ELSE 0 END) AS maduro_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'madurez.pt'
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN verde_count > maduro_count THEN 1 ELSE 0 END) AS mangos_verdes,
            SUM(CASE WHEN maduro_count > verde_count THEN 1 ELSE 0 END) AS mangos_maduros
        FROM
            item_votes;
        """
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()

        mangos_verdes = result[0] if result and result[0] is not None else 0
        mangos_maduros = result[1] if result and result[1] is not None else 0
        total_mangos = mangos_verdes + mangos_maduros
        
        return round((mangos_maduros / total_mangos) * 100, 2) if total_mangos > 0 else 0.0

    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        return 0.0
    finally:
        conn.close()

def get_porcentaje_mangos_con_defecto_lote(lote_number):
    """
    Retorna el porcentaje de mangos únicos clasificados con defecto
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_con_defectos' THEN 1 ELSE 0 END) AS con_defectos_count,
                SUM(CASE WHEN detection_type = 'mango_sin_defectos' THEN 1 ELSE 0 END) AS sin_defectos_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'defectos.pt'
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN con_defectos_count > sin_defectos_count THEN 1 ELSE 0 END) AS mangos_con_defecto,
            SUM(CASE WHEN sin_defectos_count > con_defectos_count THEN 1 ELSE 0 END) AS mangos_sin_defecto
        FROM
            item_votes;
        """
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()

        mangos_con_defecto = result[0] if result and result[0] is not None else 0
        mangos_sin_defecto = result[1] if result and result[1] is not None else 0
        total_mangos = mangos_con_defecto + mangos_sin_defecto
        
        return round((mangos_con_defecto / total_mangos) * 100, 2) if total_mangos > 0 else 0.0

    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        return 0.0
    finally:
        conn.close()

def get_porcentaje_mangos_sin_defecto_lote(lote_number):
    """
    Retorna el porcentaje de mangos únicos clasificados sin defecto
    para un lote dado, usando la lógica de mayoría de votos por item_id.
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    try:
        query = """
        WITH item_votes AS (
            SELECT
                item_id,
                SUM(CASE WHEN detection_type = 'mango_con_defectos' THEN 1 ELSE 0 END) AS con_defectos_count,
                SUM(CASE WHEN detection_type = 'mango_sin_defectos' THEN 1 ELSE 0 END) AS sin_defectos_count
            FROM
                detections
            WHERE
                lote_number = ? AND model_name = 'defectos.pt'
            GROUP BY
                item_id
        )
        SELECT
            SUM(CASE WHEN con_defectos_count > sin_defectos_count THEN 1 ELSE 0 END) AS mangos_con_defecto,
            SUM(CASE WHEN sin_defectos_count > con_defectos_count THEN 1 ELSE 0 END) AS mangos_sin_defecto
        FROM
            item_votes;
        """
        cursor.execute(query, (int(lote_number),))
        result = cursor.fetchone()

        mangos_con_defecto = result[0] if result and result[0] is not None else 0
        mangos_sin_defecto = result[1] if result and result[1] is not None else 0
        total_mangos = mangos_con_defecto + mangos_sin_defecto
        
        return round((mangos_sin_defecto / total_mangos) * 100, 2) if total_mangos > 0 else 0.0

    except sqlite3.Error as e:
        print(f"Error en la base de datos: {e}")
        return 0.0
    finally:
        conn.close()

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

    # Se revisa si todas las detecciones son 'no detections'
    if all(r == 'no detections' for r in resultados):
        return 'Nulo'

    # Se filtran las detecciones que no son 'no detections'
    detecciones_validas = [r for r in resultados if r != 'no detections']
    
    if not detecciones_validas:
        return 'Sin datos suficientes'

    # Se cuentan las detecciones válidas
    exportable = detecciones_validas.count('exportable')
    no_exportable = detecciones_validas.count('no_exportable')

    if no_exportable >= exportable:
        return 'No Exportable'
    else:
        return 'Exportable'

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

    # Se revisa si todas las detecciones son 'no detections'
    if all(r == 'no detections' for r in resultados):
        return 'Nulo'

    # Se filtran las detecciones que no son 'no detections'
    detecciones_validas = [r for r in resultados if r != 'no detections']
    
    if not detecciones_validas:
        return 'Sin datos suficientes'

    # Se cuentan las detecciones válidas
    verde = detecciones_validas.count('mango_verde')
    maduro = detecciones_validas.count('mango_maduro')

    if maduro >= verde:
        return 'Maduro'
    else:
        return 'Verde'

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

    # Se revisa si todas las detecciones son 'no detections'
    if all(r == 'no detections' for r in resultados):
        return 'Nulo'

    # Se filtran las detecciones que no son 'no detections'
    detecciones_validas = [r for r in resultados if r != 'no detections']
    
    if not detecciones_validas:
        return 'Sin datos suficientes'

    # Se cuentan las detecciones válidas
    sin_defectos = detecciones_validas.count('mango_sin_defectos')
    con_defectos = detecciones_validas.count('mango_con_defectos')

    if con_defectos >= sin_defectos:
        return 'Si'
    else:
        return 'No'

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