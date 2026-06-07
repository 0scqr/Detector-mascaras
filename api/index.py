import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Intentar cargar tflite_runtime primero (compatible con Vercel serverless),
# luego ai_edge_litert (local), finalmente tensorflow.lite como fallback
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            raise ImportError("No se encontró el intérprete de TensorFlow Lite. Instala tflite-runtime, ai-edge-litert o tensorflow.")

app = Flask(__name__)
CORS(app) # Habilitar CORS para pruebas locales

# Ruta absoluta del modelo TFLite
# En Vercel, __file__ está en /var/task/api/index.py, el modelo está en /var/task/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'mask_detector_trained.tflite')

# Fallback: buscar en directorio actual si BASE_DIR no funciona
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mask_detector_trained.tflite')
    MODEL_PATH = os.path.normpath(MODEL_PATH)

# Inicializar Intérprete TFLite
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo TFLite en {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]['dtype']
input_quant = input_details[0].get('quantization', (0, 0))
input_scale, input_zero_point = input_quant if input_quant != (0, 0) else (1.0, 0)
print(f"[Modelo] dtype entrada: {input_dtype}, cuantización: escala={input_scale}, cero={input_zero_point}")
print(f"[Modelo] forma entrada: {input_details[0]['shape']}")
print(f"[Modelo] dtype salida: {output_details[0]['dtype']}")

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
            
            # Preprocesar rostro para el modelo: RGB, 224x224, normalizar a [0,1]
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            
            # Si el modelo espera entrada cuantizada (uint8), convertir
            if input_dtype == np.uint8:
                face_img = (face_img / input_scale + input_zero_point).astype(np.uint8)
            
            # Correr inferencia TFLite
            interpreter.set_tensor(input_details[0]['index'], face_img)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # Interpretar predicción (umbral 0.7 para reducir falsos positivos)
            if pred > 0.7:
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
