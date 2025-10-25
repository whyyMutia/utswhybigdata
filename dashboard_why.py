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
    # Model deteksi objek YOLO
    yolo_model = YOLO("model/whymutia_laporan4.pt")

    # Model klasifikasi TFLite
    interpreter = tf.lite.Interpreter(model_path="model/whymutia_laporan2.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return yolo_model, interpreter, input_details, output_details


yolo_model, interpreter, input_details, output_details = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        img_array = img_array / 255.0

        # 🔹 Prediksi dengan model TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        class_index = np.argmax(prediction)
        class_names = ['Kelas 1', 'Kelas 2', 'Kelas 3']  # ganti dengan nama kelas model kamu

        st.write("### Hasil Prediksi:", class_names[class_index] if class_index < len(class_names) else class_index)
        st.write("Probabilitas:", float(np.max(prediction)))
