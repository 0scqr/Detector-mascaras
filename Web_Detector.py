import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuración de página de Streamlit
st.set_page_config(page_title="Detector de Máscaras", page_layout="centered")
st.title("😷 Detector de Máscaras Faciales en la Web")
st.write("Sube una foto o usa tu cámara para verificar si llevas mascarilla.")

# Cargar el clasificador de rostros y el modelo entrenado
@st.cache_resource
def init_models():
    import os
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    model_path = 'mask_detector_trained.h5'
    # Reensamblar modelo si no existe pero los fragmentos sí están
    if not os.path.exists(model_path):
        parts = [f"{model_path}.part1", f"{model_path}.part2", f"{model_path}.part3"]
        if all(os.path.exists(p) for p in parts):
            with open(model_path, "wb") as outfile:
                for p in parts:
                    with open(p, "rb") as infile:
                        outfile.write(infile.read())
                        
    model = load_model(model_path)
    return face_cascade, model

try:
    faceCascade, model = init_models()
    st.success("✓ Modelos cargados correctamente")
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# Entrada de cámara en la web
img_file = st.camera_input("Toma una foto con tu cámara")

if img_file is not None:
    # Convertir la imagen capturada a un array de OpenCV (BGR)
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Procesamiento de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    if len(faces) == 0:
        st.warning("No se detectaron rostros. Por favor, asegúrate de estar frente a la cámara.")
    else:
        for (x, y, w, h) in faces:
            # Extraer región del rostro
            face_roi = frame[y:y+h, x:x+w]
            
            # Convertir a RGB y preprocesar (sin dividir por 255.0)
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype("float32")
            face_img = np.expand_dims(face_img, axis=0)
            
            # Predecir
            pred = model.predict(face_img, verbose=0)[0][0]
            
            # Interpretar
            if pred > 0.5:
                label = f"CON MASCARA ({(pred * 100):.1f}%)"
                color = (0, 255, 0)  # Verde
            else:
                label = f"SIN MASCARA ({((1 - pred) * 100):.1f}%)"
                color = (0, 0, 255)  # Rojo
            
            # Dibujar rectángulos en la imagen BGR original
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convertir a RGB para mostrar en Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Resultado de la Detección", use_column_width=True)
