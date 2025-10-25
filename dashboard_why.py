import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Model YOLO (khusus bunga)
    yolo_model = YOLO("model/whymutia_laporan4.pt")

    # Model TFLite (khusus hewan)
    tflite_interpreter = tf.lite.Interpreter(model_path="model/whymutia_laporan2.tflite")
    tflite_interpreter.allocate_tensors()

    return yolo_model, tflite_interpreter

yolo_model, tflite_interpreter = load_models()

# ==========================
# Label untuk tiap model
# ==========================
animal_classes = ["Kucing", "Anjing", "Satwa Liar"]
flower_classes = ["Daisy", "Dandelion"]

# ==========================
# UI
# ==========================
st.title("🌸🐾 Deteksi Hewan & Bunga")

menu = st.sidebar.selectbox("Pilih Mode:", ["Klasifikasi Hewan (TFLite)", "Deteksi Bunga (YOLO)"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE 1: KLASIFIKASI HEWAN (TFLITE)
    # ==========================
    if menu == "Klasifikasi Hewan (TFLite)":
        # Preprocessing gambar
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        tflite_interpreter.set_tensor(input_details[0]['index'], img_array)
        tflite_interpreter.invoke()
        prediction = tflite_interpreter.get_tensor(output_details[0]['index'])[0]

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_class = animal_classes[class_index]

        # Logika sederhana: kalau akurasi terlalu kecil, asumsikan bukan hewan
        if confidence < 0.5:
            st.warning("🚫 Gambar ini tidak dikenali sebagai hewan. Silakan unggah gambar hewan (kucing, anjing, atau satwa liar).")
        else:
            st.success(f"🐾 Ini adalah **{predicted_class}** dengan probabilitas {confidence:.2%}")

    # ==========================
    # MODE 2: DETEKSI BUNGA (YOLO)
    # ==========================
    elif menu == "Deteksi Bunga (YOLO)":
        results = yolo_model(img)
        boxes = results[0].boxes

        # Filter hasil berdasarkan confidence dan label valid
        valid_detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])

                # Hanya terima label 0 atau 1 (Daisy/Dandelion) dengan confidence >= 0.6
                if cls_idx in [0, 1] and conf >= 0.6:
                    valid_detections.append((flower_classes[cls_idx], conf))

        if not valid_detections:
            st.error("🚫 Tidak terdeteksi bunga Daisy atau Dandelion. Silakan unggah gambar bunga.")
        else:
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi Bunga", use_container_width=True)
            for label, conf in valid_detections:
                st.success(f"🌸 Terdeteksi: **{label}** ({conf:.2%})")
