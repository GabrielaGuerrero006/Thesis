###############################################################
# app.py
# -------------------------------------------------------------
# Sistema de Clasificación de Mangos Ataulfo
# -------------------------------------------------------------
# Este archivo implementa la lógica principal del backend Flask
# para el sistema de clasificación de mangos usando modelos YOLO.
# Permite la captura de video, detección automática por etapas,
# almacenamiento de resultados y visualización de análisis.
# -------------------------------------------------------------
###############################################################
from flask import Flask, render_template, Response, jsonify, send_from_directory
from ultralytics import YOLO
import os
import cv2
import time
import random
import datetime
import threading
import sqlite3
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
    save_image_db # Importar la nueva función para guardar imágenes
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

# Control de lotes e IDs
used_lote_numbers = set()  # Números de lote ya usados
used_id_numbers = set()    # Números de ID ya usados
current_lote = None        # Lote actual
current_id = None          # ID actual
detections_buffer = []     # Buffer para almacenar todas las detecciones antes de guardar
photo_taken_for_current_id = False # Controla si ya se tomó la foto para el ID actual


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
    global camera, camera_running, output_frame, detection_start_time, overall_detection_start_time, model_stage, current_model, photo_taken_for_current_id, current_lote, current_id

    try:
        model_stage = 0
        print("DEBUG: Inicia el thread de generación de frames.")

        # Definir los tiempos de duración para cada etapa del modelo
        duration_stage1 = 7  # Segundos para el modelo de exportabilidad
        duration_stage_others = 5 # Segundos para los modelos de madurez y defectos
        photo_capture_time = 10 # Tiempo en segundos para tomar la foto, relativo al inicio general

        while camera_running:
            if not camera or not camera.isOpened():
                print("ERROR: La cámara no está disponible o se cerró inesperadamente.")
                stop_detection()
                break

            current_time = time.time()
            
            # Tiempo transcurrido para la etapa actual del modelo
            # Asegurarse de que detection_start_time no sea None antes de usarlo
            elapsed_time_current_stage = current_time - (detection_start_time if detection_start_time is not None else current_time)
            # Tiempo transcurrido desde el inicio general de la detección
            elapsed_time_overall = current_time - (overall_detection_start_time if overall_detection_start_time is not None else current_time)

            # Lógica para tomar la foto
            if not photo_taken_for_current_id and elapsed_time_overall >= photo_capture_time:
                print(f"DEBUG: Condición para tomar foto cumplida. Tiempo total: {elapsed_time_overall:.2f}s. photo_taken_for_current_id: {photo_taken_for_current_id}")
                success_frame, frame_to_save = camera.read()
                if success_frame:
                    image_dir = os.path.join('images', str(current_lote))
                    os.makedirs(image_dir, exist_ok=True) # Asegurarse de que el directorio exista
                    image_filename = f"{current_lote}-{current_id}.jpg"
                    image_path = os.path.join(image_dir, image_filename)
                    cv2.imwrite(image_path, frame_to_save)
                    save_image_db(current_lote, current_id, image_path) # Guardar en la nueva tabla
                    photo_taken_for_current_id = True
                    print(f"DEBUG: Foto capturada y guardada en disco y DB: {image_path}")
                else:
                    print("ADVERTENCIA: No se pudo capturar el frame para guardar la foto.")

            if model_stage == 0:
                model_stage = 1
                current_model = YOLO('exportabilidad.pt')
                detection_start_time = time.time() # Establecer el inicio para esta etapa
                print("DEBUG: Cargando modelo de exportabilidad.")

            elif model_stage == 1 and elapsed_time_current_stage >= duration_stage1 and camera_running:
                print(f"DEBUG: Cambiando a modelo de madurez. Tiempo transcurrido en etapa: {elapsed_time_current_stage:.2f}s")
                model_stage = 2
                current_model = YOLO('madurez.pt')
                detection_start_time = current_time # Reiniciar el tiempo para la nueva etapa

            elif model_stage == 2 and elapsed_time_current_stage >= duration_stage_others and camera_running:
                print(f"DEBUG: Cambiando a modelo de defectos. Tiempo transcurrido en etapa: {elapsed_time_current_stage:.2f}s")
                model_stage = 3
                current_model = YOLO('defectos.pt')
                detection_start_time = current_time # Reiniciar el tiempo para la nueva etapa

            elif model_stage == 3 and elapsed_time_current_stage >= duration_stage_others and camera_running:
                print(f"DEBUG: Finalizando detección automáticamente. Tiempo transcurrido en etapa: {elapsed_time_current_stage:.2f}s")
                model_stage = 4 # Marcar como finalizado antes de detener
                stop_detection()
                break # Salir del bucle while

            if not camera_running:
                print("DEBUG: camera_running es False, saliendo del bucle de frames.")
                break

            success, frame = camera.read()
            if not success:
                print("ERROR: Error al leer frame de la cámara principal. Deteniendo detección.")
                stop_detection()
                break

            if current_model:
                try:
                    results = current_model.predict(frame, conf=0.85)
                    
                    # Determinar el nombre del modelo actual
                    modelo_nombre = ""
                    if model_stage == 1:
                        modelo_nombre = "exportabilidad.pt"
                    elif model_stage == 2:
                        modelo_nombre = "madurez.pt"
                    elif model_stage == 3:
                        modelo_nombre = "defectos.pt"
                    
                    # Procesar resultados y añadir al buffer
                    detections = process_results(results, modelo_nombre)
                    
                    annotated_frame = results[0].plot()

                    modelo_texto = ""
                    if model_stage == 1:
                        modelo_texto = "Modelo: exportabilidad.pt"
                    elif model_stage == 2:
                        modelo_texto = "Modelo: madurez.pt"
                    elif model_stage == 3:
                        modelo_texto = "Modelo: defectos.pt"

                    # Calcular el tiempo restante basado en la etapa actual
                    tiempo_restante = 0
                    if model_stage == 1:
                        tiempo_restante = duration_stage1 - elapsed_time_current_stage
                    elif model_stage in [2, 3]:
                        tiempo_restante = duration_stage_others - elapsed_time_current_stage
                    
                    # Asegurarse de que tiempo_restante no sea negativo para la visualización
                    tiempo_restante = max(0, tiempo_restante)

                    # Mostrar información del lote y ID en el frame
                    cv2.putText(annotated_frame, f"Lote: {current_lote} | ID: {current_id}",
                                 (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{modelo_texto} - Tiempo restante: {tiempo_restante:.1f}s",
                                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)