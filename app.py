# app.py
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import threading
import time
import csv
import os
import datetime
import random

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
csv_file_path = "detecciones.csv"

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

def save_detections_to_csv():
    """Guarda todas las detecciones del buffer al archivo CSV"""
    global detections_buffer
    
    if not detections_buffer:
        print("No hay detecciones para guardar")
        return
    
    # Crear el archivo CSV con las cabeceras si no existe
    file_exists = os.path.exists(csv_file_path)
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir cabeceras si el archivo es nuevo
        if not file_exists:
            writer.writerow(['Lote', 'ID', 'Fecha', 'Hora', 'Modelo', 'Tipo_Deteccion', 'Confianza'])
        
        # Escribir todas las detecciones del buffer
        for detection in detections_buffer:
            writer.writerow(detection)
    
    print(f"Guardadas {len(detections_buffer)} detecciones en {csv_file_path}")
    # Limpiar el buffer después de guardar
    detections_buffer = []

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

@app.route('/stop_and_save')
def stop_and_save():
    global current_lote
    try:
        # Detener la cámara
        stop_detection()
        release_camera()
        
        # Guardar las detecciones en CSV
        save_detections_to_csv()
        
        # Resetear el lote para la próxima sesión
        current_lote = None
        
        return jsonify({
            "status": "success", 
            "message": f"Detecciones guardadas exitosamente. Total de registros guardados: {len(detections_buffer) if detections_buffer else 0}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error al guardar: {str(e)}"})

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)