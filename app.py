from flask import Flask, render_template, Response, jsonify, send_from_directory
from ultralytics import YOLO
import os
import cv2
import time
import random
import datetime
import threading
import sqlite3
import serial
from database import (
    init_db, save_detections_db, get_lotes, get_ids_lote,
    get_num_mangos_procesados, get_num_detecciones_lote, get_num_exportables_no_exportables,
    get_num_verdes_maduros, get_num_con_defectos_sin_defectos, get_fecha_procesado_lote,
    get_confianza_promedio_lote, get_confianza_promedio_exportabilidad, get_confianza_promedio_madurez,
    get_confianza_promedio_defectos, get_cantidad_mangos_exportables_lote, get_cantidad_mangos_no_exportables_lote,
    get_cantidad_mangos_verdes_lote, get_cantidad_mangos_maduros_lote, get_cantidad_mangos_con_defecto_lote,
    get_cantidad_mangos_sin_defecto_lote, get_porcentaje_mangos_exportables_lote, get_porcentaje_mangos_no_exportables_lote,
    get_porcentaje_mangos_verdes_lote, get_porcentaje_mangos_maduros_lote, get_porcentaje_mangos_con_defecto_lote,
    get_porcentaje_mangos_sin_defecto_lote, get_ids_lote,
    # NUEVAS FUNCIONES PARA ANÁLISIS POR ID
    get_fecha_deteccion_lote_id, get_exportabilidad_mango, get_madurez_mango, get_defectos_mango,
    get_confianza_promedio_exportabilidad_mango, get_confianza_promedio_madurez_mango, get_confianza_promedio_defectos_mango,
    save_image_db, get_images_by_lote_and_id # Importar la nueva función para guardar imágenes
)
from images import (
    generar_grafico_exportables_pie,
    generar_grafico_verdes_maduros_pie,
    generar_grafico_con_sin_defectos_pie,
    generar_grafico_confianza_promedio_bar
)

# Inicializar la base de datos al inicio
init_db()

app = Flask(__name__)


# ----------------------
# Variables globales
# ----------------------
# Control de la cámara y detección
camera = None
camera_running = False
output_frame = None
lock = threading.Lock()
detection_thread = None
detection_start_time = None # Tiempo de inicio para la etapa actual del modelo
overall_detection_start_time = None # Tiempo de inicio para toda la detección
model_stage = 0 # 0: no iniciado, 1: exportabilidad, 2: madurez, 3: defectos, 4: finalizado
current_model = None
arduino_serial = None

# Control de lotes e IDs
used_lote_numbers = set()  # Números de lote ya usados
used_id_numbers = set()    # Números de ID ya usados
current_lote = None        # Lote actual
current_id = None          # ID actual
detections_buffer = []     # Buffer para almacenar todas las detecciones antes de guardar
photo_taken_for_current_id = False # Controla si ya se tomó la foto para el ID actual

# NUEVAS: Variables globales para los nombres de los modelos
modelo_nombre_para_global_buffer = "" # Para add_detection_to_buffer (usa .pt)
current_model_name_for_local_analysis = "" # Para analyze_and_send_signals_to_arduino (sin .pt)


# ----------------------
# Funciones auxiliares
# ----------------------
def generate_unique_number(used_set):
    """
    Genera un número único de 5 dígitos que no se haya usado antes.
    Se utiliza para lotes e IDs de mangos.
    """
    while True:
        number = random.randint(10000, 99999)
        if number not in used_set:
            used_set.add(number)
            return number

def generate_lote():
    """
    Genera un nuevo código de lote único y lo asigna como lote actual.
    """
    global current_lote
    current_lote = generate_unique_number(used_lote_numbers)
    return current_lote

def generate_id():
    """
    Genera un nuevo código de ID único y lo asigna como ID actual.
    También reinicia el control de foto tomada para el nuevo ID.
    """
    global current_id, photo_taken_for_current_id
    current_id = generate_unique_number(used_id_numbers)
    photo_taken_for_current_id = False # Resetear al generar un nuevo ID
    print(f"DEBUG: Nuevo ID generado: {current_id}. photo_taken_for_current_id reseteado a False.")
    return current_id

def setup_arduino_serial(port='COM3', baudrate=9600): # Ajusta el puerto COM según tu Arduino (ej. 'COM3' en Windows, '/dev/ttyACM0' en Linux)
    """
    Inicializa la conexión serial con Arduino.
    """
    global arduino_serial
    try:
        if arduino_serial is None or not arduino_serial.is_open:
            arduino_serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Da tiempo a Arduino para reiniciarse después de abrir el puerto serial
            print(f"DEBUG: Conexión serial con Arduino establecida en {port} a {baudrate} baudios.")
    except serial.SerialException as e:
        print(f"ERROR: No se pudo establecer conexión serial con Arduino: {e}")
        arduino_serial = None

def send_arduino_signal(pin, state):
    """
    Envía una señal a Arduino para un pin específico y estado (H para HIGH, L para LOW).
    Ejemplo: send_arduino_signal(7, 'H')
    """
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        try:
            signal = f"{pin}{state}\n".encode('utf-8') # Añade un salto de línea para facilitar el parsing en Arduino
            arduino_serial.write(signal)
            print(f"DEBUG: Enviado '{signal.decode().strip()}' a Arduino.")
        except Exception as e:
            print(f"ERROR: No se pudo enviar señal a Arduino: {e}")
    else:
        print("ADVERTENCIA: Conexión serial con Arduino no establecida.")

def analyze_and_send_signals_to_arduino(detections_list, lote, item_id):
    """
    Analiza las detecciones de YOLO para un lote e ID específicos y envía señales a Arduino.
    Esta función ahora recibe directamente las detecciones del mango actual.
    detections_list: Una lista de listas, donde cada sublista es [lote, ID, fecha, hora, modelo, clase, confianza].
    """
    print(f"DEBUG: Iniciando análisis de detecciones para Lote: {lote}, ID: {item_id}")

    # Inicializa contadores para las clases relevantes de cada modelo
    counts = {
        'exportabilidad': {'exportable': 0, 'no_exportable': 0},
        'madurez': {'verde': 0, 'maduro': 0},
        'defectos': {'con_defecto': 0, 'sin_defecto': 0}
    }

    # Filtra y cuenta las detecciones
    for det in detections_list:
        _lote, _item_id, _date, _time, model_name, class_name, _confidence = det
        
        # Ensure the model_name is processed correctly from the simplified local analysis name
        if model_name == 'exportabilidad':
            if class_name == 'exportable':
                counts['exportabilidad']['exportable'] += 1
            elif class_name == 'no_exportable':
                counts['exportabilidad']['no_exportable'] += 1
        elif model_name == 'madurez':
            if class_name == 'mango_verde': # Assuming 'mango_verde' corresponds to 'verde'
                counts['madurez']['verde'] += 1
            elif class_name == 'mango_maduro': # Assuming 'mango_maduro' corresponds to 'maduro'
                counts['madurez']['maduro'] += 1
        elif model_name == 'defectos':
            if class_name == 'mango_con_defectos': # Assuming 'mango_con_defectos' corresponds to 'con_defecto'
                counts['defectos']['con_defecto'] += 1
            elif class_name == 'mango_sin_defectos': # Assuming 'mango_sin_defectos' corresponds to 'sin_defecto'
                counts['defectos']['sin_defecto'] += 1
        elif class_name == 'no detections': # Handle cases where no objects were detected
            pass
        else:
            print(f"ADVERTENCIA: Clase '{class_name}' no esperada para el modelo '{model_name}'.")

    print(f"DEBUG: Recuento de detecciones: {counts}")

    # Lógica de decisión para el Pin 12 (Exportable)
    # Condiciones para exportable:
    # 1. exportable > no_exportable
    # 2. verde > maduro
    # 3. sin_defecto > con_defecto
    is_exportable_candidate = (
        counts['exportabilidad']['exportable'] > counts['exportabilidad']['no_exportable'] and
        counts['madurez']['verde'] > counts['madurez']['maduro'] and
        counts['defectos']['sin_defecto'] > counts['defectos']['con_defecto']
    )

    # Lógica de decisión para el Pin 13 (No Exportable)
    # Prioritizamos la lógica de Pin 12. Si no es exportable, por defecto se activa Pin 13.
    if is_exportable_candidate:
        send_arduino_signal(13, 'L') # Asegurarse de que Pin 13 esté en LOW
        send_arduino_signal(12, 'H') # Activar Pin 12
        print("DECISION: Mango probablemente exportable. Señal HIGH en Pin 12.")
        threading.Timer(5, send_arduino_signal, args=(12, 'L')).start() # Desactivar después de 5 segundos
    else:
        send_arduino_signal(12, 'L') # Asegurarse de que Pin 12 esté en LOW
        send_arduino_signal(13, 'H') # Activar Pin 13 como no exportable
        print("DECISION: Mango probablemente NO exportable. Señal HIGH en Pin 13.")
        threading.Timer(5, send_arduino_signal, args=(13, 'L')).start() # Desactivar después de 5 segundos


def save_detections_to_db():
    """
    Guarda todas las detecciones almacenadas en el buffer en la base de datos.
    El buffer se limpia después de guardar exitosamente.
    """
    global detections_buffer, current_lote
    if not detections_buffer:
        print("No hay detecciones para guardar")
        return
    if current_lote is None:
        raise ValueError("No hay un lote activo para guardar las detecciones")
    try:
        save_detections_db(detections_buffer)
        print(f"Guardadas {len(detections_buffer)} detecciones en la base de datos")
        detections_buffer = []
    except Exception as e:
        print(f"Error al guardar las detecciones: {e}")
        raise


def add_detection_to_buffer(model_name, detections):
    """
    Añade los resultados de detección al buffer en memoria.
    Cada entrada incluye lote, ID, fecha, hora, modelo, clase detectada y confianza.
    """
    global detections_buffer, current_lote, current_id
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')
    time_str = current_time.strftime('%H:%M:%S')
    if len(detections) == 0:
        # No se detectaron objetos
        detections_buffer.append([current_lote, current_id, date_str, time_str, model_name, 'no detections', 0.0])
    else:
        # Guardar cada detección con su clase y confianza
        for det in detections:
            class_name = det[0]
            confidence = det[1]
            detections_buffer.append([current_lote, current_id, date_str, time_str, model_name, class_name, confidence])


def init_camera():
    """
    Inicializa la cámara web para la captura de video.
    Configura la resolución y retorna el objeto cámara.
    """
    global camera
    try:
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("No se pudo abrir la cámara")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return camera
    except Exception as e:
        print(f"Error al inicializar la cámara: {str(e)}")
        return None


def stop_detection():
    """
    Detiene la detección y reinicia variables de control.
    """
    global camera_running, model_stage, current_model, output_frame
    print("Deteniendo detección...")
    camera_running = False
    model_stage = 0
    current_model = None
    output_frame = None


def release_camera():
    """
    Libera los recursos de la cámara y detiene el thread de detección si está activo.
    """
    global camera, detection_thread, camera_running
    print("Liberando recursos de la cámara...")
    if camera is not None:
        camera.release()
        camera = None
    if detection_thread and detection_thread.is_alive() and threading.current_thread() != detection_thread:
        detection_thread.join(timeout=5)
        detection_thread = None
    camera_running = False
    print("Cámara y thread de detección detenidos.")


def process_results(results, model_name):
    """
    Procesa los resultados de YOLO y devuelve una lista de detecciones.
    Cada detección es una tupla (nombre_clase, confianza).
    Además, añade los resultados al buffer para su posterior guardado.
    """
    detections = []
    # Si hay detecciones
    if results[0].boxes and len(results[0].boxes.cls) > 0:
        # Recorrer cada detección
        for i in range(len(results[0].boxes.cls)):
            class_id = int(results[0].boxes.cls[i].item())
            confidence = float(results[0].boxes.conf[i].item())
            class_name = results[0].names[class_id]
            detections.append((class_name, confidence))
    # Añadir al buffer en lugar de guardar directamente
    add_detection_to_buffer(model_name, detections)
    return detections

def generate_frames_thread():
    global camera, camera_running, output_frame, detection_start_time, overall_detection_start_time, model_stage, current_model, photo_taken_for_current_id, current_lote, current_id, modelo_nombre_para_global_buffer, current_model_name_for_local_analysis

    try:
        model_stage = 0
        print("DEBUG: Inicia el thread de generación de frames.")

        # Definir los tiempos de duración para cada etapa del modelo
        duration_stage1 = 7  # Segundos para el modelo de exportabilidad
        duration_stage_others = 5 # Segundos para los modelos de madurez y defectos
        total_processing_duration = duration_stage1 + (2 * duration_stage_others) # 7 + 5 + 5 = 17 seconds

        # Tiempos para tomar las 4 fotos (en segundos desde el inicio)
        photo_capture_times = [4, 8, 12, 16]
        photos_taken = [False, False, False, False]  # Controla si ya se tomó cada foto

        # NUEVA: Lista local para almacenar detecciones del mango actual
        current_mango_detections_local = []
        
        # Mantener un registro del ID del mango que se está procesando actualmente en la lista local
        local_processing_mango_id = None 

        while camera_running:
            if not camera or not camera.isOpened():
                print("ERROR: La cámara no está disponible o se cerró inesperadamente.")
                stop_detection()
                break

            current_time = time.time()

            # Tiempo transcurrido para la etapa actual del modelo
            elapsed_time_current_stage = current_time - (detection_start_time if detection_start_time is not None else current_time)
            # Tiempo transcurrido desde el inicio general de la detección
            elapsed_time_overall = current_time - (overall_detection_start_time if overall_detection_start_time is not None else current_time)
            
            # Si el current_id global ha cambiado, significa que un nuevo mango está comenzando.
            # Esto resetea el buffer local y establece el ID de procesamiento local.
            # Esto ocurrirá cuando se presione "Iniciar" nuevamente.
            if local_processing_mango_id != current_id:
                if len(current_mango_detections_local) > 0:
                    # Si había detecciones para un mango anterior que no fue procesado explícitamente
                    print(f"ADVERTENCIA: Mango {local_processing_mango_id} no fue analizado localmente antes de cambiar a {current_id}. Analizando ahora.")
                    # Ensure Pin 7 is LOW before starting analysis for the previous mango, if it wasn't already.
                    send_arduino_signal(7, 'L')
                    analyze_and_send_signals_to_arduino(current_mango_detections_local, current_lote, local_processing_mango_id)
                current_mango_detections_local = []
                local_processing_mango_id = current_id
                print(f"DEBUG: Nuevo mango ({current_id}) detectado, reiniciando buffer local de detecciones.")
                # Resetear el control de fotos para el nuevo mango
                photos_taken = [False, False, False, False]
                # NEW: Send HIGH to Pin 7 when a new mango's processing cycle starts
                send_arduino_signal(7, 'H')
                print("DEBUG: Signal HIGH to Pin 7 (detection started for new mango).")


            # Lógica para detener el proceso después de que haya transcurrido el tiempo total de procesamiento.
            # Esto asegura que el análisis final se realice y luego el sistema se detenga.
            if overall_detection_start_time is not None and elapsed_time_overall >= total_processing_duration:
                print(f"DEBUG: Tiempo total de procesamiento ({total_processing_duration}s) transcurrido para mango ID: {local_processing_mango_id}. Finalizando ciclo de detección.")
                
                # NEW: Send LOW to Pin 7 before analysis and stopping
                send_arduino_signal(7, 'L')
                print("DEBUG: Signal LOW to Pin 7 (detection finished for this mango).")
                time.sleep(1)

                # Realizar análisis inmediato para el mango actual antes de detener
                if len(current_mango_detections_local) > 0:
                    print(f"DEBUG: Mango {local_processing_mango_id} procesado completamente por los 3 modelos. Iniciando análisis local de Arduino.")
                    analyze_and_send_signals_to_arduino(current_mango_detections_local, current_lote, local_processing_mango_id)
                else:
                    print(f"DEBUG: No se detectaron objetos para el mango {local_processing_mango_id} a lo largo de las etapas de los modelos (antes de detener).")

                # Limpiar el buffer local
                current_mango_detections_local = []
                local_processing_mango_id = None 
                
                stop_detection() # Esto establecerá camera_running en False
                print("DEBUG: Detección detenida por tiempo total transcurrido.")
                break # Salir del bucle while para terminar el thread

            # Lógica para tomar las 4 fotos en los tiempos definidos
            for idx, capture_time in enumerate(photo_capture_times):
                if not photos_taken[idx] and overall_detection_start_time is not None and elapsed_time_overall >= capture_time:
                    print(f"DEBUG: Condición para tomar foto {idx+1} cumplida. Tiempo total: {elapsed_time_overall:.2f}s.")
                    success_frame, frame_to_save = camera.read()
                    if success_frame:
                        image_dir = os.path.join('images', str(current_lote))
                        os.makedirs(image_dir, exist_ok=True) # Asegurarse de que el directorio exista
                        image_filename = f"{current_lote}-{current_id}-{idx+1}.jpg"
                        image_path = os.path.join(image_dir, image_filename)
                        cv2.imwrite(image_path, frame_to_save)
                        # Guardar en la base de datos como BLOB además de la ruta
                        save_image_db(current_lote, current_id, image_path)
                        photos_taken[idx] = True
                        print(f"DEBUG: Foto {idx+1} capturada y guardada en disco y DB: {image_path}")
                    else:
                        print(f"ADVERTENCIA: No se pudo capturar el frame para guardar la foto {idx+1}.")

            # Transiciones de etapa del modelo y asignación de current_model
            # Estas variables son globales y se asignan solo en las transiciones de etapa
            # NO deben inicializarse a "" en cada iteración del bucle.

            if model_stage == 0:
                model_stage = 1
                current_model = YOLO('exportabilidad.pt')
                detection_start_time = time.time() # Establecer el inicio para esta etapa
                # overall_detection_start_time ya se inicializa en start_camera para el inicio del ciclo.
                # Asegurarse de que no sea None si el thread se inicia de alguna otra forma (safety check)
                if overall_detection_start_time is None:
                    overall_detection_start_time = time.time() 
                print("DEBUG: Cargando modelo de exportabilidad.")
                # Asignar los valores a las variables globales
                modelo_nombre_para_global_buffer = "exportabilidad.pt"
                current_model_name_for_local_analysis = "exportabilidad"

            elif model_stage == 1 and elapsed_time_current_stage >= duration_stage1:
                print(f"DEBUG: Cambiando a modelo de madurez. Tiempo transcurrido en etapa: {elapsed_time_current_stage:.2f}s")
                model_stage = 2
                current_model = YOLO('madurez.pt')
                detection_start_time = current_time # Reiniciar el tiempo para la nueva etapa
                # Asignar los valores a las variables globales
                modelo_nombre_para_global_buffer = "madurez.pt"
                current_model_name_for_local_analysis = "madurez"

            elif model_stage == 2 and elapsed_time_current_stage >= duration_stage_others:
                print(f"DEBUG: Cambiando a modelo de defectos. Tiempo transcurrido en etapa: {elapsed_time_current_stage:.2f}s")
                model_stage = 3
                current_model = YOLO('defectos.pt')
                detection_start_time = current_time # Reiniciar el tiempo para la nueva etapa
                # Asignar los valores a las variables globales
                modelo_nombre_para_global_buffer = "defectos.pt"
                current_model_name_for_local_analysis = "defectos"

            # Nota: El bloque anterior para `model_stage == 3` que reiniciaba el ciclo
            # ha sido eliminado. Ahora, una vez que el modelo de defectos termina su tiempo,
            # el control pasará a la verificación de `total_processing_duration` al inicio del bucle
            # para detener el sistema.

            if not camera_running:
                print("DEBUG: camera_running es False, saliendo del bucle de frames (después de transiciones de modelo).")
                break

            success, frame = camera.read()
            if not success:
                print("ERROR: Error al leer frame de la cámara principal. Deteniendo detección.")
                stop_detection()
                break

            if current_model:
                try:
                    results = current_model.predict(frame, conf=0.85)
                    
                    # Procesar resultados y añadir al buffer global
                    detections_from_model = process_results(results, modelo_nombre_para_global_buffer) 

                    # ADICIÓN: Rellenar current_mango_detections_local con el nombre simplificado del modelo
                    current_time_for_detection = datetime.datetime.now()
                    date_str_for_detection = current_time_for_detection.strftime('%Y-%m-%d')
                    time_str_for_detection = current_time_for_detection.strftime('%H:%M:%S')

                    if len(detections_from_model) > 0:
                        for det_class_name, det_confidence in detections_from_model:
                            current_mango_detections_local.append([current_lote, current_id, date_str_for_detection, time_str_for_detection, current_model_name_for_local_analysis, det_class_name, det_confidence])
                            
                    annotated_frame = results[0].plot()

                    # Texto para la visualización en el frame
                    modelo_texto = "" 
                    if model_stage == 1:
                        modelo_texto = "Modelo: exportabilidad.pt"
                    elif model_stage == 2:
                        modelo_texto = "Modelo: madurez.pt"
                    elif model_stage == 3:
                        modelo_texto = "Modelo: defectos.pt"

                    # Calcular el tiempo restante basado en la etapa actual
                    tiempo_restante_etapa = 0
                    if model_stage == 1:
                        tiempo_restante_etapa = duration_stage1 - elapsed_time_current_stage
                    elif model_stage in [2, 3]:
                        tiempo_restante_etapa = duration_stage_others - elapsed_time_current_stage
                    
                    # Asegurarse de que tiempo_restante_etapa no sea negativo para la visualización
                    tiempo_restante_etapa = max(0, tiempo_restante_etapa)

                    # Tiempo total restante para la detección completa (hasta los 17 segundos)
                    tiempo_total_restante = total_processing_duration - elapsed_time_overall
                    tiempo_total_restante = max(0, tiempo_total_restante)

                    # Mostrar información del lote y ID en el frame
                    cv2.putText(annotated_frame, f"Lote: {current_lote} | ID: {current_id}",
                                 (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{modelo_texto} - Etapa Restante: {tiempo_restante_etapa:.1f}s",
                                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Total Restante: {tiempo_total_restante:.1f}s",
                                 (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


                    with lock:
                        ret, buffer = cv2.imencode('.jpg', annotated_frame)
                        if ret:
                            output_frame = buffer.tobytes()
                except Exception as e:
                    print(f"ERROR: Error en la predicción del modelo: {str(e)}")
                    continue
            else:
                with lock:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        output_frame = buffer.tobytes()

            time.sleep(0.03) # Pequeña pausa para no saturar la CPU

    except Exception as e:
        print(f"ERROR: Error crítico en generate_frames_thread: {str(e)}")
        stop_detection()
    finally:
        print("DEBUG: Thread de detección finalizado (finally block).")

def generate():
    global output_frame, camera_running, lock
    while camera_running:
        with lock:
            if output_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n') # Frame vacío si no hay nada
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera, camera_running, model_stage, current_model, detection_thread, current_lote, current_id, photo_taken_for_current_id, overall_detection_start_time, detection_start_time
    # Asegúrate de incluir las nuevas variables globales aquí también, si se inicializan al inicio
    global modelo_nombre_para_global_buffer, current_model_name_for_local_analysis
    
    try:
        if camera_running:
            return jsonify({"status": "warning", "message": "La cámara ya está en funcionamiento"})
        
        # Asegurarnos de que los recursos anteriores estén liberados
        release_camera()
        
        camera = init_camera()
        if camera is None:
            return jsonify({"status": "error", "message": "No se pudo inicializar la cámara"})
        
        # Generar códigos: nuevo lote si no existe, siempre nuevo ID
        if current_lote is None:
            generate_lote()
            print(f"DEBUG: Nuevo lote generado: {current_lote}")
        
        generate_id() # Generar un nuevo ID y resetear photo_taken_for_current_id
        
        camera_running = True
        model_stage = 0
        current_model = None
        overall_detection_start_time = time.time() # Establecer el tiempo de inicio general
        detection_start_time = time.time() # Inicializar detection_start_time aquí también

        # Inicializar los nombres de los modelos para el primer ciclo del mango
        # Esto es importante para que tengan un valor desde el principio
        modelo_nombre_para_global_buffer = "" 
        current_model_name_for_local_analysis = ""

        detection_thread = threading.Thread(target=generate_frames_thread)
        detection_thread.start()
        
        return jsonify({
            "status": "success", 
            "message": "Cámara iniciada",
            "lote": current_lote,
            "id": current_id
        })
    except Exception as e:
        print(f"ERROR: Error al iniciar la cámara: {str(e)}")
        return jsonify({"status": "error", "message": f"Error al iniciar la cámara: {str(e)}"})

@app.route('/stop_camera')
def stop_camera():
    stop_detection() # Primero detenemos la detección
    release_camera() # Luego liberamos la cámara
    return jsonify({"status": "success", "message": "Cámara detenida"})

@app.route('/save_detections')
def save_detections():
    global current_lote, detections_buffer
    try:
        if not detections_buffer:
            return jsonify({
                "status": "error",
                "message": "No hay detecciones para guardar"
            })
        
        if current_lote is None:
            return jsonify({
                "status": "error",
                "message": "No hay un lote activo"
            })
        
        # Contar detecciones antes de guardar
        detections_count = len(detections_buffer)
        
        # Guardar las detecciones en la base de datos
        save_detections_to_db()
        
        # Resetear el lote para la próxima sesión
        current_lote = None
        
        return jsonify({
            "status": "success",
            "message": f"Se han guardado {detections_count} detecciones exitosamente"
        })
        
    except Exception as e:
        print(f"ERROR: Error al guardar las detecciones: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error al guardar las detecciones: {str(e)}"
        })

@app.route('/camera_status')
def camera_status():
    global camera_running, model_stage, current_lote, current_id
    return jsonify({
        "running": camera_running,
        "model_stage": model_stage,
        "lote": current_lote,
        "id": current_id,
        "detections_count": len(detections_buffer)
    })

@app.route('/obtener_lotes')
def obtener_lotes():
    try:
        lotes = get_lotes()
        return jsonify({"status": "success", "lotes": lotes})
    except Exception as e:
        print(f"ERROR: Error al obtener lotes: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_ids_lote/<lote_number>')
def obtener_ids_lote(lote_number):
    try:
        ids = get_ids_lote(lote_number)
        return jsonify({"status": "success", "ids": ids})
    except Exception as e:
        print(f"ERROR: Error al obtener IDs de lote: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_datos_lote/<lote_number>')
def obtener_datos_lote(lote_number):
    try:
        # Inicializar un diccionario para agrupar los datos por categoría
        datos_agrupados = {
            "Datos generales": [],
            "Exportabilidad": [],
            "Madurez": [],
            "Defectos": [],
            "Confianza": []
        }

        # Datos Generales
        datos_agrupados["Datos generales"].append(["Fecha del lote", get_fecha_procesado_lote(lote_number)])
        datos_agrupados["Datos generales"].append(["Numero de frames del lote", get_num_detecciones_lote(lote_number)])
        datos_agrupados["Datos generales"].append(["Numero de mangos en el lote", get_num_mangos_procesados(lote_number)])

        # Exportabilidad
        exportables = get_num_exportables_no_exportables(lote_number)
        datos_agrupados["Exportabilidad"].append(["Numero de frames de mango exportable", exportables['exportable']])
        datos_agrupados["Exportabilidad"].append(["Numero de frames de mango no exportable", exportables['no_exportable']])
        datos_agrupados["Exportabilidad"].append(["Cantidad de mangos exportables del lote", get_cantidad_mangos_exportables_lote(lote_number)])
        datos_agrupados["Exportabilidad"].append(["Cantidad de mangos no exportables del lote", get_cantidad_mangos_no_exportables_lote(lote_number)])
        datos_agrupados["Exportabilidad"].append(["Porcentaje de mangos exportables del lote", f"{get_porcentaje_mangos_exportables_lote(lote_number)}%"])
        datos_agrupados["Exportabilidad"].append(["Porcentaje de mangos no exportables del lote", f"{get_porcentaje_mangos_no_exportables_lote(lote_number)}%"])

        # Madurez
        maduros_verdes = get_num_verdes_maduros(lote_number)
        datos_agrupados["Madurez"].append(["Numero de frames de mango verde", maduros_verdes['mango_verde']])
        datos_agrupados["Madurez"].append(["Numero de frames de mango maduro", maduros_verdes['mango_maduro']])
        datos_agrupados["Madurez"].append(["Cantidad de mangos verdes del lote", get_cantidad_mangos_verdes_lote(lote_number)])
        datos_agrupados["Madurez"].append(["Cantidad de mangos maduros del lote", get_cantidad_mangos_maduros_lote(lote_number)])
        datos_agrupados["Madurez"].append(["Porcentaje de mangos verdes del lote", f"{get_porcentaje_mangos_verdes_lote(lote_number)}%"])
        datos_agrupados["Madurez"].append(["Porcentaje de mangos maduros del lote", f"{get_porcentaje_mangos_maduros_lote(lote_number)}%"])

        # Defectos
        defectos = get_num_con_defectos_sin_defectos(lote_number)
        datos_agrupados["Defectos"].append(["Numero de frames de mango sin defectos", defectos['mango_sin_defectos']])
        datos_agrupados["Defectos"].append(["Numero de frames de mango con defectos", defectos['mango_con_defectos']])
        datos_agrupados["Defectos"].append(["Cantidad de mangos sin defectos del lote", get_cantidad_mangos_sin_defecto_lote(lote_number)])
        datos_agrupados["Defectos"].append(["Cantidad de mangos con defectos del lote", get_cantidad_mangos_con_defecto_lote(lote_number)])
        datos_agrupados["Defectos"].append(["Porcentaje de mangos sin defectos del lote", f"{get_porcentaje_mangos_sin_defecto_lote(lote_number)}%"])
        datos_agrupados["Defectos"].append(["Porcentaje de mangos con defectos del lote", f"{get_porcentaje_mangos_con_defecto_lote(lote_number)}%"])

        # Confianza promedio
        datos_agrupados["Confianza"].append(["Porcentaje de confianza promedio de todos los modelos del lote", f"{get_confianza_promedio_lote(lote_number)}%"])
        datos_agrupados["Confianza"].append(["Porcentaje de confianza promedio del modelo exportabilidad", f"{get_confianza_promedio_exportabilidad(lote_number)}%"])
        datos_agrupados["Confianza"].append(["Porcentaje de confianza promedio del modelo madurez", f"{get_confianza_promedio_madurez(lote_number)}%"])
        datos_agrupados["Confianza"].append(["Porcentaje de confianza promedio del modelo defectos", f"{get_confianza_promedio_defectos(lote_number)}%"])
        
        return jsonify({"status": "success", "datos": datos_agrupados})
    except Exception as e:
        print(f"ERROR: Error al obtener datos del lote: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_datos_por_id/<lote_number>/<id_number>')
def obtener_datos_por_id(lote_number, id_number):
    try:
        # Obtener datos generales del lote
        datos_lote = {}
        datos_lote["Numero de mangos en el lote"] = get_num_mangos_procesados(lote_number)
        datos_lote["Numero de frames del lote"] = get_num_detecciones_lote(lote_number)
        exportables = get_num_exportables_no_exportables(lote_number)
        datos_lote["Numero de frames de mango exportable"] = exportables['exportable']
        datos_lote["Numero de frames de mango no exportable"] = exportables['no_exportable']
        maduros_verdes = get_num_verdes_maduros(lote_number)
        datos_lote["Numero de frames de mango maduro"] = maduros_verdes['mango_maduro']
        datos_lote["Numero de frames de mango verde"] = maduros_verdes['mango_verde']
        defectos = get_num_con_defectos_sin_defectos(lote_number)
        datos_lote["Numero de frames de mango con defectos"] = defectos['mango_con_defectos']
        datos_lote["Numero de frames de mango sin defectos"] = defectos['mango_sin_defectos']
        datos_lote["Fecha del lote"] = get_fecha_procesado_lote(lote_number)
        datos_lote["Porcentaje de confianza promedio de todos los modelos del lote"] = f"{get_confianza_promedio_lote(lote_number)}%"
        datos_lote["Porcentaje de confianza promedio del modelo exportabilidad"] = f"{get_confianza_promedio_exportabilidad(lote_number)}%"
        datos_lote["Porcentaje de confianza promedio del modelo madurez"] = f"{get_confianza_promedio_madurez(lote_number)}%"
        datos_lote["Porcentaje de confianza promedio del modelo defectos"] = f"{get_confianza_promedio_defectos(lote_number)}%"
        datos_lote["Cantidad de mangos exportables del lote"] = get_cantidad_mangos_exportables_lote(lote_number)
        datos_lote["Cantidad de mangos no exportables del lote"] = get_cantidad_mangos_no_exportables_lote(lote_number)
        datos_lote["Cantidad de mangos verdes del lote"] = get_cantidad_mangos_verdes_lote(lote_number)
        datos_lote["Cantidad de mangos maduros del lote"] = get_cantidad_mangos_maduros_lote(lote_number)
        datos_lote["Cantidad de mangos con defectos del lote"] = get_cantidad_mangos_con_defecto_lote(lote_number)
        datos_lote["Cantidad de mangos sin defectos del lote"] = get_cantidad_mangos_sin_defecto_lote(lote_number)
        datos_lote["Porcentaje de mangos exportables del lote"] = f"{get_porcentaje_mangos_exportables_lote(lote_number)}%"
        datos_lote["Porcentaje de mangos no exportables del lote"] = f"{get_porcentaje_mangos_no_exportables_lote(lote_number)}%"
        datos_lote["Porcentaje de mangos verdes del lote"] = f"{get_porcentaje_mangos_verdes_lote(lote_number)}%"
        datos_lote["Porcentaje de mangos maduros del lote"] = f"{get_porcentaje_mangos_maduros_lote(lote_number)}%"
        datos_lote["Porcentaje de mangos con defectos del lote"] = f"{get_porcentaje_mangos_con_defecto_lote(lote_number)}%"
        datos_lote["Porcentaje de mangos sin defectos del lote"] = f"{get_porcentaje_mangos_sin_defecto_lote(lote_number)}%"

        # Obtener datos específicos por ID
        datos_id = []
        datos_id.append(["Fecha de detección", get_fecha_deteccion_lote_id(lote_number, id_number)])
        datos_id.append(["Exportabilidad", get_exportabilidad_mango(lote_number, id_number)])
        datos_id.append(["Madurez", get_madurez_mango(lote_number, id_number)])
        datos_id.append(["Defectos", get_defectos_mango(lote_number, id_number)])
        datos_id.append(["Confianza promedio exportabilidad", f"{get_confianza_promedio_exportabilidad_mango(lote_number, id_number)}%"])
        datos_id.append(["Confianza promedio madurez", f"{get_confianza_promedio_madurez_mango(lote_number, id_number)}%"])
        datos_id.append(["Confianza promedio defectos", f"{get_confianza_promedio_defectos_mango(lote_number, id_number)}%"])

        return jsonify({"status": "success", "datos_lote": datos_lote, "datos_id": datos_id})
    except Exception as e:
        print(f"ERROR: Error al obtener datos por ID: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_datos_mango/<lote_number>/<item_id>')
def obtener_datos_mango(lote_number, item_id):
    try:
        # *** CAMBIO AQUÍ: CONSTRUIR LA LISTA DE LISTAS CON EL ORDEN ESPECIFICADO ***
        datos = []
        datos.append(["Fecha de detección", get_fecha_deteccion_lote_id(lote_number, item_id)])
        datos.append(["Exportabilidad", get_exportabilidad_mango(lote_number, item_id)])
        datos.append(["Madurez", get_madurez_mango(lote_number, item_id)])
        datos.append(["Defectos", get_defectos_mango(lote_number, item_id)])
        datos.append(["Confianza promedio exportabilidad", f"{get_confianza_promedio_exportabilidad_mango(lote_number, item_id)}%"])
        datos.append(["Confianza promedio madurez", f"{get_confianza_promedio_madurez_mango(lote_number, item_id)}%"])
        datos.append(["Confianza promedio defectos", f"{get_confianza_promedio_defectos_mango(lote_number, item_id)}%"])
        
        return jsonify({"status": "success", "datos": datos})
    except Exception as e:
        print(f"ERROR: Error al obtener datos del mango: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/generar_imagenes_lote/<lote_number>', methods=['POST'])
def generar_imagenes_lote(lote_number):
    try:
        generar_grafico_exportables_pie(lote_number)
        generar_grafico_verdes_maduros_pie(lote_number)
        generar_grafico_con_sin_defectos_pie(lote_number)
        generar_grafico_confianza_promedio_bar(lote_number)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"ERROR: Error al generar imágenes del lote: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/imagenes_lote/<lote_number>/<filename>')
def imagenes_lote(lote_number, filename):
    # Sirve archivos de imagen de la carpeta images/<lote_number>
    import os
    dir_path = os.path.join('images', str(lote_number))
    return send_from_directory(dir_path, filename)

@app.route('/obtener_rutas_imagenes_lote/<lote_number>')
def obtener_rutas_imagenes_lote(lote_number):
    try:
        # Devuelve las rutas relativas de las imágenes generadas para el lote
        # Mapeo de nombres de archivo a claves descriptivas para el frontend
        mapa_imagenes = {
            'Exportables-NoExportables-Pie.jpg': 'exportabilidad_img',
            'Verdes-Maduros-Pie.jpg': 'madurez_img',
            'Con-Sin-Defectos-Pie.jpg': 'defectos_img',
            'Confianza-Promedio-Bar.jpg': 'confianza_img'
        }
        
        rutas_por_categoria = {}
        import os
        base_dir = os.path.join('images', str(lote_number))

        for filename, key in mapa_imagenes.items():
            full_path = os.path.join(base_dir, filename)
            if os.path.exists(full_path):
                rutas_por_categoria[key] = f'/imagenes_lote/{lote_number}/{filename}'
            else:
                rutas_por_categoria[key] = None # O un string vacío, o un placeholder si se desea

        # *** CAMBIO CLAVE AQUÍ: Asegurarse de incluir "status": "success" ***
        return jsonify({"status": "success", "imagenes": rutas_por_categoria})
    except Exception as e:
        print(f"ERROR: Error en obtener_rutas_imagenes_lote: {e}") # Log para depuración
        return jsonify({"status": "error", "message": str(e)})

# Nueva ruta para obtener imágenes por lote e ID (galería por ID)
@app.route('/obtener_imagenes_mango/<lote_number>/<item_id>')
def obtener_imagenes_mango(lote_number, item_id):
    try:
        imagenes = get_images_by_lote_and_id(lote_number, item_id)
        return jsonify({
            "status": "success",
            "imagenes": imagenes
        })
    except Exception as e:
        print(f"ERROR: Error al obtener imágenes por ID: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error al obtener imágenes: {str(e)}"
        })

if __name__ == '__main__':
    setup_arduino_serial()
    app.run(debug=True, host='0.0.0.0', port=5000)