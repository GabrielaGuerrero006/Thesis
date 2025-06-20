from database import get_lote_data, get_lotes, get_ids_for_lote

def cargar_lote(numero_lote, directorio=None):
    """
    Carga los datos de un lote específico desde la base de datos.
    
    Args:
        numero_lote (str): Número del lote a cargar (ej: "95383")
        directorio (str): Parámetro mantenido por compatibilidad, no se utiliza
    
    Returns:
        pandas.DataFrame: DataFrame con los datos del lote específico
        None: Si hay un error al cargar el lote
    """
    try:
        return get_lote_data(numero_lote)
    except Exception as e:
        print(f"Error al cargar el lote {numero_lote}: {e}")
        return None

def obtener_numeros_lote(directorio=None):
    """
    Obtiene solo los números de lote de la base de datos.
    
    Args:
        directorio (str): Parámetro mantenido por compatibilidad, no se utiliza
        
    Returns:
        list: Lista con los números de lote encontrados
    """
    return get_lotes()

def obtener_ids_unicos(numero_lote, directorio=None):
    """
    Obtiene la cantidad y lista de IDs únicos para un lote específico.
    
    Args:
        numero_lote (str): Número del lote a analizar (ej: "95383")
        directorio (str): Parámetro mantenido por compatibilidad, no se utiliza
    
    Returns:
        tuple: (cantidad_ids, lista_ids)
            - cantidad_ids (int): Número de IDs únicos encontrados
            - lista_ids (list): Lista con todos los IDs únicos
        None: Si hay error al cargar el lote
    """
    try:
        return get_ids_for_lote(numero_lote)
    except Exception as e:
        print(f"Error al obtener IDs únicos para el lote {numero_lote}: {e}")
        return None, None