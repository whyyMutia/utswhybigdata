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
st.title("🌸🐾 Deteksi Bunga & Klasifikasi Hewan")
menu = st.sidebar.radio("Pilih Model yang Akan Dijalankan:", ["Klasifikasi Hewan (CNN)", "Deteksi Bunga (YOLO)"])

# ==========================
# KONFIRMASI USER
# ==========================
if menu == "Klasifikasi Hewan (CNN)":
    st.subheader("📘 Mode: Klasifikasi Hewan (TFLite)")
    confirm = st.radio("Apakah kamu yakin gambar yang ingin diunggah adalah **hewan**?", ["Ya", "Tidak"])

    if confirm == "Tidak":
        st.warning("⚠️ Pastikan ulang bahwa gambar kamu adalah hewan sebelum mengunggah!")
    else:
        uploaded_file = st.file_uploader("Unggah gambar hewan di sini 🐾", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Gambar yang Diupload", use_container_width=True)

            # Preprocessing gambar
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            # Prediksi
            tflite_interpreter.set_tensor(input_details[0]['index'], img_array)
            tflite_interpreter.invoke()
            prediction = tflite_interpreter.get_tensor(output_details[0]['index'])[0]

            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence < 0.7:
                st.error("🚫 Gambar ini tidak dikenali sebagai hewan.")
            else:
                st.success(f"🐾 Hasil: **{animal_classes[class_index]}** ({confidence:.2%})")

# =====================================================
# YOLO BAGIAN DETEKSI BUNGA
# =====================================================
elif menu == "Deteksi Bunga (YOLO)":
    st.subheader("🌼 Mode: Deteksi Bunga (YOLO)")
    confirm = st.radio("Apakah kamu yakin gambar yang ingin diunggah adalah **bunga**?", ["Ya", "Tidak"])

    if confirm == "Tidak":
        st.warning("⚠️ Pastikan ulang bahwa gambar kamu adalah bunga sebelum mengunggah!")
    else:
        uploaded_file = st.file_uploader("Unggah gambar bunga di sini 🌸", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Gambar yang Diupload", use_container_width=True)

            # Deteksi objek dengan YOLO
            results = yolo_model(img)
            boxes = results[0].boxes

            valid_detections = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_idx in [0, 1] and conf >= 0.6:
                        valid_detections.append((flower_classes[cls_idx], conf))

            if not valid_detections:
                st.error("🚫 Tidak terdeteksi bunga Daisy atau Dandelion. Pastikan gambar yang diunggah adalah bunga.")
            else:
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi Bunga", use_container_width=True)
                for label, conf in valid_detections:
                    st.success(f"🌸 Terdeteksi: **{label}** ({conf:.2%})")
