import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Intentar cargar tflite_runtime (ideal para Vercel) o fallback a tensorflow.lite (local)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("No se encontró el intérprete de TensorFlow Lite. Instala tflite-runtime o tensorflow.")

app = Flask(__name__)
CORS(app) # Habilitar CORS para pruebas locales

# Ruta absoluta del modelo TFLite
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'mask_detector_trained.tflite')

# Inicializar Intérprete TFLite
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo TFLite en {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ruta al clasificador Haar Cascade incluido en el proyecto
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Fallback: usar el integrado en OpenCV si no se encuentra en el proyecto
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


@app.route('/api/detect', methods=['POST'])
def detect_mask():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        # Procesar la imagen base64
        image_b64 = data['image']
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
            
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400
            
        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extraer ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocesar rostro para el modelo (RGB, 224x224, float32)
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype("float32")
            face_img = np.expand_dims(face_img, axis=0)
            
            # Correr inferencia TFLite
            interpreter.set_tensor(input_details[0]['index'], face_img)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # Interpretar predicción
            if pred > 0.5:
                label = "CON MASCARA"
                confidence = float(pred * 100)
            else:
                label = "SIN MASCARA"
                confidence = float((1 - pred) * 100)
                
            results.append({
                "box": [int(x), int(y), int(w), int(h)],
                "label": label,
                "confidence": confidence,
                "raw_pred": float(pred)
            })
            
        return jsonify({
            "success": True,
            "faces": results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_size_mb": os.path.getsize(MODEL_PATH) / (1024 * 1024) if os.path.exists(MODEL_PATH) else 0
    })

# Ruta comodín para que Vercel sirva Flask en /api/
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({"success": False, "error": f"API Route not found: {path}"}), 404

if __name__ == '__main__':
    print("Iniciando servidor de desarrollo en http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
