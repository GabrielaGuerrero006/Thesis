# app.py
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
    get_confianza_promedio_exportabilidad_mango, get_confianza_promedio_madurez_mango, get_confianza_promedio_defectos_mango
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

# Variables globales para el control de la cámara
camera = None
camera_running = False
output_frame = None
lock = threading.Lock()
detection_thread = None
detection_start_time = None
model_stage = 0  # 0: no iniciado, 1: exportabilidad, 2: madurez, 3: defectos, 4: finalizado
current_model = None
# Removido csv_file_path estático - ahora se genera dinámicamente

# Variables para el sistema de lotes e IDs
used_lote_numbers = set()
used_id_numbers = set()
current_lote = None
current_id = None
detections_buffer = []  # Buffer para almacenar todas las detecciones antes de guardar

def generate_unique_number(used_set):
    """Genera un número único de 5 dígitos que no se haya usado antes"""
    while True:
        number = random.randint(10000, 99999)
        if number not in used_set:
            used_set.add(number)
            return number

def generate_lote():
    """Genera un nuevo código de lote único"""
    global current_lote
    current_lote = generate_unique_number(used_lote_numbers)
    return current_lote

def generate_id():
    """Genera un nuevo código de ID único"""
    global current_id
    current_id = generate_unique_number(used_id_numbers)
    return current_id



def save_detections_to_db():
    """Guarda todas las detecciones del buffer a la base de datos"""
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
    """Añade los resultados de detección al buffer en memoria"""
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
    global camera_running, model_stage, current_model, output_frame
    print("Deteniendo detección...")
    camera_running = False
    model_stage = 0
    current_model = None
    output_frame = None

def release_camera():
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
    """Procesa los resultados de YOLO y devuelve lista de detecciones"""
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
    global camera, camera_running, output_frame, detection_start_time, model_stage, current_model

    try:
        model_stage = 0
        print("Iniciando el thread de generación de frames.")

        while camera_running:
            if not camera or not camera.isOpened():
                print("Error: La cámara no está disponible")
                stop_detection()
                break

            if model_stage == 0:
                model_stage = 1
                current_model = YOLO('exportabilidad.pt')
                detection_start_time = time.time()
                print("Cargando modelo de exportabilidad")

            current_time = time.time()
            elapsed_time = current_time - detection_start_time

            if model_stage == 1 and elapsed_time >= 5 and camera_running:
                model_stage = 2
                current_model = YOLO('madurez.pt')
                detection_start_time = current_time
                print("Cargando modelo de madurez")

            elif model_stage == 2 and elapsed_time >= 5 and camera_running:
                model_stage = 3
                current_model = YOLO('defectos.pt')
                detection_start_time = current_time
                print("Cargando modelo de defectos")

            elif model_stage == 3 and elapsed_time >= 5 and camera_running:
                model_stage = 4
                print("Finalizando detección automáticamente desde el thread.")
                stop_detection()
                break

            if not camera_running:
                print("Thread de generación de frames finalizado.")
                break

            success, frame = camera.read()
            if not success:
                print("Error al leer frame de la cámara")
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

                    tiempo_restante = 5 - elapsed_time if model_stage in [1, 2, 3] else 0
                    
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
                    print(f"Error en la predicción del modelo: {str(e)}")
                    continue
            else:
                with lock:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        output_frame = buffer.tobytes()

            time.sleep(0.03)

    except Exception as e:
        print(f"Error en generate_frames_thread: {str(e)}")
        stop_detection()
    finally:
        print("Thread de detección finalizado")

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
    global camera, camera_running, model_stage, current_model, detection_thread, current_lote, current_id
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
            print(f"Nuevo lote generado: {current_lote}")
        
        generate_id()
        print(f"Nuevo ID generado: {current_id}")
        
        camera_running = True
        model_stage = 0
        current_model = None
        detection_thread = threading.Thread(target=generate_frames_thread)
        detection_thread.start()
        
        return jsonify({
            "status": "success", 
            "message": "Cámara iniciada",
            "lote": current_lote,
            "id": current_id
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error al iniciar la cámara: {str(e)}"})

@app.route('/stop_camera')
def stop_camera():
    stop_detection()  # Primero detenemos la detección
    release_camera()  # Luego liberamos la cámara
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
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_ids_lote/<lote_number>')
def obtener_ids_lote(lote_number):
    try:
        ids = get_ids_lote(lote_number)
        return jsonify({"status": "success", "ids": ids})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_datos_lote/<lote_number>')
def obtener_datos_lote(lote_number):
    try:
        datos = {}
        datos["Numero de mangos en el lote"] = get_num_mangos_procesados(lote_number)
        datos["Numero de detecciones del lote"] = get_num_detecciones_lote(lote_number)
        exportables = get_num_exportables_no_exportables(lote_number)
        datos["Numero de detecciones de mango exportable"] = exportables['exportable']
        datos["Numero de detecciones de mango no_exportable"] = exportables['no_exportable']
        maduros_verdes = get_num_verdes_maduros(lote_number)
        datos["Numero de detecciones de mango maduro"] = maduros_verdes['mango_maduro']
        datos["Numero de detecciones de mango verde"] = maduros_verdes['mango_verde']
        defectos = get_num_con_defectos_sin_defectos(lote_number)
        datos["Numero de detecciones de mango con_defectos"] = defectos['mango_con_defectos']
        datos["Numero de detecciones de mango sin_defectos"] = defectos['mango_sin_defectos']
        datos["Fecha del lote"] = get_fecha_procesado_lote(lote_number)
        datos["Porcentaje de confianza promedio de todos los modelos del lote"] = f"{get_confianza_promedio_lote(lote_number)}%"
        datos["Porcentaje de confianza promedio del modelo exportabilidad"] = f"{get_confianza_promedio_exportabilidad(lote_number)}%"
        datos["Porcentaje de confianza promedio del modelo madurez"] = f"{get_confianza_promedio_madurez(lote_number)}%"
        datos["Porcentaje de confianza promedio del modelo defectos"] = f"{get_confianza_promedio_defectos(lote_number)}%"
        datos["Cantidad de mangos exportables del lote"] = get_cantidad_mangos_exportables_lote(lote_number)
        datos["Cantidad de mangos no exportables del lote"] = get_cantidad_mangos_no_exportables_lote(lote_number)
        datos["Cantidad de mangos verdes del lote"] = get_cantidad_mangos_verdes_lote(lote_number)
        datos["Cantidad de mangos maduros del lote"] = get_cantidad_mangos_maduros_lote(lote_number)
        datos["Cantidad de mangos con defectos del lote"] = get_cantidad_mangos_con_defecto_lote(lote_number)
        datos["Cantidad de mangos sin defectos del lote"] = get_cantidad_mangos_sin_defecto_lote(lote_number)
        datos["Porcentaje de mangos exportables del lote"] = f"{get_porcentaje_mangos_exportables_lote(lote_number)}%"
        datos["Porcentaje de mangos no exportables del lote"] = f"{get_porcentaje_mangos_no_exportables_lote(lote_number)}%"
        datos["Porcentaje de mangos verdes del lote"] = f"{get_porcentaje_mangos_verdes_lote(lote_number)}%"
        datos["Porcentaje de mangos maduros del lote"] = f"{get_porcentaje_mangos_maduros_lote(lote_number)}%"
        datos["Porcentaje de mangos con defectos del lote"] = f"{get_porcentaje_mangos_con_defecto_lote(lote_number)}%"
        datos["Porcentaje de mangos sin defectos del lote"] = f"{get_porcentaje_mangos_sin_defecto_lote(lote_number)}%"
        return jsonify({"status": "success", "datos": datos})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_datos_por_id/<lote_number>/<id_number>')
def obtener_datos_por_id(lote_number, id_number):
    try:
        # Obtener datos generales del lote
        datos_lote = {}
        datos_lote["Numero de mangos en el lote"] = get_num_mangos_procesados(lote_number)
        datos_lote["Numero de detecciones del lote"] = get_num_detecciones_lote(lote_number)
        exportables = get_num_exportables_no_exportables(lote_number)
        datos_lote["Numero de detecciones de mango exportable"] = exportables['exportable']
        datos_lote["Numero de detecciones de mango no_exportable"] = exportables['no_exportable']
        maduros_verdes = get_num_verdes_maduros(lote_number)
        datos_lote["Numero de detecciones de mango maduro"] = maduros_verdes['mango_maduro']
        datos_lote["Numero de detecciones de mango verde"] = maduros_verdes['mango_verde']
        defectos = get_num_con_defectos_sin_defectos(lote_number)
        datos_lote["Numero de detecciones de mango con_defectos"] = defectos['mango_con_defectos']
        datos_lote["Numero de detecciones de mango sin_defectos"] = defectos['mango_sin_defectos']
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
        datos_id = {}
        datos_id["Fecha de detección"] = get_fecha_deteccion_lote_id(lote_number, id_number)
        datos_id["Exportabilidad"] = get_exportabilidad_mango(lote_number, id_number)
        datos_id["Madurez"] = get_madurez_mango(lote_number, id_number)
        datos_id["Defectos"] = get_defectos_mango(lote_number, id_number)
        datos_id["Confianza promedio exportabilidad"] = get_confianza_promedio_exportabilidad_mango(lote_number, id_number)
        datos_id["Confianza promedio madurez"] = get_confianza_promedio_madurez_mango(lote_number, id_number)
        datos_id["Confianza promedio defectos"] = get_confianza_promedio_defectos_mango(lote_number, id_number)

        return jsonify({"status": "success", "datos_lote": datos_lote, "datos_id": datos_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obtener_datos_mango/<lote_number>/<item_id>')
def obtener_datos_mango(lote_number, item_id):
    try:
        datos = {}
        datos["Fecha de detección"] = get_fecha_deteccion_lote_id(lote_number, item_id)
        datos["Exportabilidad"] = get_exportabilidad_mango(lote_number, item_id)
        datos["Madurez"] = get_madurez_mango(lote_number, item_id)
        datos["Defectos"] = get_defectos_mango(lote_number, item_id)
        datos["Confianza promedio exportabilidad"] = f"{get_confianza_promedio_exportabilidad_mango(lote_number, item_id)}%"
        datos["Confianza promedio madurez"] = f"{get_confianza_promedio_madurez_mango(lote_number, item_id)}%"
        datos["Confianza promedio defectos"] = f"{get_confianza_promedio_defectos_mango(lote_number, item_id)}%"
        return jsonify({"status": "success", "datos": datos})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generar_imagenes_lote/<lote_number>', methods=['POST'])
def generar_imagenes_lote(lote_number):
    try:
        generar_grafico_exportables_pie(lote_number)
        generar_grafico_verdes_maduros_pie(lote_number)
        generar_grafico_con_sin_defectos_pie(lote_number)
        generar_grafico_confianza_promedio_bar(lote_number)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/imagenes_lote/<lote_number>/<filename>')
def imagenes_lote(lote_number, filename):
    # Sirve archivos de imagen de la carpeta images/<lote_number>
    import os
    dir_path = os.path.join('images', str(lote_number))
    return send_from_directory(dir_path, filename)

@app.route('/obtener_rutas_imagenes_lote/<lote_number>')
def obtener_rutas_imagenes_lote(lote_number):
    # Devuelve las rutas relativas de las imágenes generadas para el lote
    nombres = [
        'Exportables-NoExportables-Pie.jpg',
        'Verdes-Maduros-Pie.jpg',
        'Con-Sin-Defectos-Pie.jpg',
        'Confianza-Promedio-Bar.jpg'
    ]
    rutas = [f'/imagenes_lote/{lote_number}/{nombre}' for nombre in nombres]
    # Solo incluir las que existan
    import os
    base_dir = os.path.join('images', str(lote_number))
    rutas_existentes = [ruta for ruta, nombre in zip(rutas, nombres) if os.path.exists(os.path.join(base_dir, nombre))]
    return jsonify({"imagenes": rutas_existentes})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)