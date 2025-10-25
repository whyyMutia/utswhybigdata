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
    # Model YOLO untuk deteksi bunga
    yolo_model = YOLO("model/whymutia_laporan4.pt")

    # Model klasifikasi TFLite untuk hewan
    interpreter = tf.lite.Interpreter(model_path="model/whymutia_laporan2.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return yolo_model, interpreter, input_details, output_details


yolo_model, interpreter, input_details, output_details = load_models()

# ==========================
# UI
# ==========================
st.title("🌸🐾 Smart Image Detector & Classifier")
st.write("Aplikasi ini dapat mengenali **bunga (daisy, dandelion)** dan **hewan (kucing, anjing, atau wild animal)** secara otomatis.")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# =========================
# Prediction Logic
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # --- Step 1: klasifikasi hewan dengan TFLite ---
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_array = img_array / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    class_index = np.argmax(prediction)
    class_names = ['Kucing', 'Anjing', 'Hewan Liar / Wild']  # nama kelas model kamu
    pred_class = class_names[class_index]
    pred_conf = float(np.max(prediction))

    st.write(f"### 🧠 Hasil Klasifikasi Hewan: {pred_class} ({pred_conf:.2f} confidence)")

    # --- Step 2: deteksi bunga dengan YOLO ---
    results = yolo_model(img)
    labels = results[0].boxes.cls.tolist() if results[0].boxes is not None else []

    # Cek apakah ada bunga daisy/dandelion terdeteksi
    flower_labels = [yolo_model.names[int(i)] for i in labels] if len(labels) > 0 else []
    if len(flower_labels) > 0:
        st.image(results[0].plot(), caption="🌼 Hasil Deteksi YOLO (Bunga)", use_container_width=True)
        st.success(f"Model YOLO mendeteksi bunga: {', '.join(flower_labels)}")
    else:
        st.warning("⚠️ Gambar ini tidak terdeteksi sebagai bunga daisy atau dandelion. Silahkan upload ulang")

    # --- Step 3: peringatan logika silang ---
    if pred_class in ['Kucing', 'Anjing', 'Hewan Liar / Wild'] and len(flower_labels) > 0:
        st.error("🚫 Gambar ini tampak seperti hewan, tetapi YOLO mendeteksi bunga. Coba periksa kembali.")
    elif pred_class in ['Kucing', 'Anjing', 'Hewan Liar / Wild'] and len(flower_labels) == 0:
        st.info("✅ Gambar ini diklasifikasikan sebagai hewan (TFLite). Tidak ada bunga terdeteksi oleh YOLO.")
    elif len(flower_labels) > 0 and pred_class == 'Wild':
        st.info("🌺 Gambar bunga berhasil terdeteksi oleh YOLO.")
