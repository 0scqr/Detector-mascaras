"""
Detector de Máscaras en Tiempo Real
Detecta rostros y clasifica si usan máscara o no.
Presiona 'Q' para salir.
"""

import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

print("=" * 60)
print("DETECTOR DE MÁSCARAS - Versión Mejorada")
print("=" * 60)

# Cargar clasificador de rostros Haar Cascade
print("\n[1/3] Cargando clasificador de rostros...")
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascade_path)
    if faceCascade.empty():
        raise Exception("No se pudo cargar el clasificador de rostros")
    print("✓ Clasificador cargado")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Cargar modelo de detección de máscaras
print("\n[2/3] Cargando modelo de detección...")
MODEL_PATH = "mask_detector_trained.h5"

# Reensamblar modelo si no existe pero los fragmentos sí están
if not os.path.exists(MODEL_PATH):
    parts = [f"{MODEL_PATH}.part1", f"{MODEL_PATH}.part2", f"{MODEL_PATH}.part3"]
    if all(os.path.exists(p) for p in parts):
        print("Reensamblando modelo de IA desde fragmentos...")
        try:
            with open(MODEL_PATH, "wb") as outfile:
                for p in parts:
                    with open(p, "rb") as infile:
                        outfile.write(infile.read())
            print("✓ Modelo reensamblado con éxito")
        except Exception as e:
            print(f"✗ Error al reensamblar el modelo: {e}")

if not os.path.exists(MODEL_PATH):
    print(f"⚠ Advertencia: '{MODEL_PATH}' no encontrado.")
    print("  Intentando usar 'mask_detector.h5'...")
    MODEL_PATH = "mask_detector.h5"
    if not os.path.exists(MODEL_PATH):
        print("\n✗ Error: No se encontró ningún modelo.")
        print("  Ejecuta 'train_mask_detector.py' primero para entrenar el modelo.\n")
        sys.exit(1)

try:
    model = load_model(MODEL_PATH)
    print(f"✓ Modelo cargado: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error al cargar modelo: {e}\n")
    sys.exit(1)

# Inicializar cámara
print("\n[3/3] Iniciando cámara...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo acceder a la cámara")
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✓ Cámara iniciada")
except Exception as e:
    print(f"✗ Error: {e}\n")
    sys.exit(1)

print("\n" + "=" * 60)
print("DETECTOR ACTIVO - Presiona 'Q' para salir")
print("=" * 60 + "\n")

# Variables para FPS
frame_count = 0
show_instructions = True

def predict_mask(face_img):
    """
    Predice si hay máscara en la imagen del rostro.
    Retorna: (probabilidad, etiqueta, color)
    """
    # Convert BGR (OpenCV) to RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Preprocesar
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    
    # Predecir
    pred = model.predict(face_img, verbose=0)[0][0]
    
    # Interpretar resultado
    # pred > 0.5 = Con máscara
    # pred < 0.5 = Sin máscara
    if pred > 0.5:
        label = "CON MASCARA"
        color = (0, 255, 0)  # Verde
        confidence = pred * 100
    else:
        label = "SIN MASCARA"
        color = (0, 0, 255)  # Rojo
        confidence = (1 - pred) * 100
        
    # Log prediction values for analysis
    print(f"[DEBUG] Prediccion raw: {pred:.4f} -> {label}")
    import datetime
    try:
        with open("predictions.log", "a") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - pred: {pred:.4f} -> {label}\n")
    except Exception:
        pass
    
    return pred, label, color, confidence

def draw_instructions(frame):
    """Dibuja instrucciones en pantalla."""
    h, w = frame.shape[:2]
    
    # Fondo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Texto
    cv2.putText(frame, "DETECTOR DE MASCARAS", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'Q' para salir | 'I' para ocultar instrucciones", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Verde = Con Mascara | Rojo = Sin Mascara", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Loop principal
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ No se pudo leer frame de la cámara")
            break
        
        frame_count += 1
        
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer región del rostro
            face_roi = frame[y:y+h, x:x+w]
            
            # Predecir
            pred, label, color, confidence = predict_mask(face_roi)
            
            # Dibujar rectángulo alrededor del rostro
            thickness = 3
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Fondo para el texto
            label_text = f"{label} ({confidence:.1f}%)"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(frame, (x, y - text_height - 15), 
                         (x + text_width + 10, y), color, -1)
            
            # Texto de etiqueta
            cv2.putText(frame, label_text, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar contador de rostros
        cv2.putText(frame, f"Rostros: {len(faces)}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar instrucciones
        if show_instructions:
            draw_instructions(frame)
        
        # Mostrar frame
        cv2.imshow('Detector de Mascaras', frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n✓ Saliendo...")
            break
        elif key == ord('i') or key == ord('I'):
            show_instructions = not show_instructions

except KeyboardInterrupt:
    print("\n✓ Detenido por el usuario")
except Exception as e:
    print(f"\n✗ Error durante la ejecución: {e}")
finally:
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Recursos liberados\n")