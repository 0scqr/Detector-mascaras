import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Rutas de archivos ────────────────────────────────────────────────────────
# En Vercel: __file__ = /var/task/api/index.py  →  BASE_DIR = /var/task/
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, 'mask_detector_trained.tflite')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# ── Variables globales (carga lazy para evitar timeout en cold-start) ────────
_interpreter   = None
_input_details = None
_output_details = None
_input_dtype   = None
_input_scale   = 1.0
_input_zero    = 0
_face_cascade  = None


def get_interpreter():
    """Carga el intérprete TFLite la primera vez que se llama (lazy loading)."""
    global _interpreter, _input_details, _output_details
    global _input_dtype, _input_scale, _input_zero

    if _interpreter is not None:
        return _interpreter

    # Importar TFLite: tensorflow-cpu incluye tf.lite
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        raise ImportError(
            "tensorflow-cpu no está instalado. "
            "Asegúrate de que requirements.txt incluye tensorflow-cpu."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")

    _interpreter = Interpreter(model_path=MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_details  = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()

    _input_dtype  = _input_details[0]['dtype']
    quant = _input_details[0].get('quantization', (0, 0))
    _input_scale, _input_zero = quant if quant != (0, 0) else (1.0, 0)

    print(f"[Modelo] dtype={_input_dtype}, shape={_input_details[0]['shape']}, "
          f"escala={_input_scale}, cero={_input_zero}")
    return _interpreter


def get_face_cascade():
    """Carga el clasificador Haar Cascade la primera vez que se llama."""
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade

    path = CASCADE_PATH
    if not os.path.exists(path):
        # Fallback al cascade integrado en OpenCV
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


# ── Rutas Flask ──────────────────────────────────────────────────────────────

@app.route('/api/detect', methods=['POST'])
def detect_mask():
    try:
        interp = get_interpreter()
        cascade = get_face_cascade()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        # Decodificar imagen base64
        image_b64 = data['image']
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Detectar rostros
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # Preprocesar: RGB, 224×224, normalizar [0,1]
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # Cuantización si el modelo lo requiere
            if _input_dtype == np.uint8:
                face_img = (face_img / _input_scale + _input_zero).astype(np.uint8)

            # Inferencia TFLite
            interp.set_tensor(_input_details[0]['index'], face_img)
            interp.invoke()
            pred = float(interp.get_tensor(_output_details[0]['index'])[0][0])

            if pred > 0.7:
                label      = "CON MASCARA"
                confidence = pred * 100
            else:
                label      = "SIN MASCARA"
                confidence = (1 - pred) * 100

            results.append({
                "box":        [int(x), int(y), int(w), int(h)],
                "label":      label,
                "confidence": round(confidence, 2),
                "raw_pred":   round(pred, 4)
            })

        return jsonify({"success": True, "faces": results})

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route('/api/health', methods=['GET'])
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status":        "healthy",
        "model_found":   model_exists,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2)
                         if model_exists else 0,
        "base_dir":      BASE_DIR,
        "model_path":    MODEL_PATH,
    })


# Ruta comodín — devuelve 404 para rutas API desconocidas
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({"success": False, "error": f"Ruta no encontrada: /{path}"}), 404


if __name__ == '__main__':
    print("Servidor de desarrollo en http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
