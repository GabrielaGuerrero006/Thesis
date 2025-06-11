import pandas as pd
import os
import glob
import re

def cargar_lote(numero_lote, directorio='detections'):
    """
    Carga los datos de un lote específico desde su archivo CSV.
    
    Args:
        numero_lote (str): Número del lote a cargar (ej: "95383")
        directorio (str): Ruta al directorio que contiene los archivos CSV de lotes.
                         Por defecto es 'detections'
    
    Returns:
        pandas.DataFrame: DataFrame con los datos del lote específico
        None: Si no se encuentra el archivo o hay un error al cargarlo
    """
    try:
        # Construir la ruta del archivo
        ruta_absoluta = os.path.abspath(directorio)
        nombre_archivo = f"Lote-{numero_lote}.csv"
        ruta_completa = os.path.join(ruta_absoluta, nombre_archivo)
        
        # Verificar si el archivo existe
        if not os.path.exists(ruta_completa):
            print(f"No se encontró el archivo para el lote {numero_lote}")
            return None
            
        # Cargar el archivo
        return pd.read_csv(ruta_completa)
        
    except Exception as e:
        print(f"Error al cargar el lote {numero_lote}: {e}")
        return None
    
def obtener_numeros_lote(directorio='detections'):
    """
    Obtiene solo los números de lote de los archivos CSV en el directorio especificado.
    
    Args:
        directorio (str): Ruta al directorio que contiene los archivos CSV de lotes.
                         Por defecto es 'detections'
        
    Returns:
        list: Lista con los números de lote encontrados
    """
    # Obtener la ruta absoluta del directorio
    ruta_absoluta = os.path.abspath(directorio)
    
    # Buscar todos los archivos CSV en el directorio
    patron_busqueda = os.path.join(ruta_absoluta, '*.csv')
    archivos_csv = glob.glob(patron_busqueda)
    
    # Lista para almacenar los números de lote
    numeros_lote = []
    
    # Patrón para extraer el número de lote del nombre del archivo
    patron_lote = re.compile(r'Lote-(\d+)\.csv')
    
    for archivo in archivos_csv:
        nombre_archivo = os.path.basename(archivo)
        match = patron_lote.match(nombre_archivo)
        if match:
            numero_lote = match.group(1)
            numeros_lote.append(numero_lote)
    
    return sorted(numeros_lote)

def obtener_ids_unicos(numero_lote, directorio='detections'):
    """
    Obtiene la cantidad y lista de IDs únicos para un lote específico.
    
    Args:
        numero_lote (str): Número del lote a analizar (ej: "95383")
        directorio (str): Ruta al directorio que contiene los archivos CSV de lotes.
                         Por defecto es 'detections'
    
    Returns:
        tuple: (cantidad_ids, lista_ids)
            - cantidad_ids (int): Número de IDs únicos encontrados
            - lista_ids (list): Lista con todos los IDs únicos
        None: Si hay error al cargar el lote
    """
    # Usar la función cargar_lote para obtener los datos
    df = cargar_lote(numero_lote, directorio)
    
    if df is not None:
        # Obtener IDs únicos
        ids_unicos = df['ID'].unique()
        return len(ids_unicos), ids_unicos.tolist()
    
    return None, None