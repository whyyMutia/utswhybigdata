import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Tampilan
# ==========================
st.set_page_config(page_title="Klasifikasi Hewan & Deteksi Bunga", layout="wide")

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
# Sidebar – Pengaturan
# ==========================
st.sidebar.header("🛠️ Pengaturan")

theme = st.sidebar.radio("Pilih Mode Tampilan:", ["🌞 Terang", "🌙 Gelap", "📖 Redup / Baca"])

# Terapkan mode tampilan (CSS dinamis)
if theme == "🌞 Terang":
    bg_color = "#FFFFFF"
    text_color = "#000000"
    sidebar_bg = "#F8F9FA"
    sidebar_text = "#000000"
elif theme == "🌙 Gelap":
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    sidebar_bg = "#0E1117"
    sidebar_text = "#FAFAFA"
elif theme == "📖 Redup / Baca":
    bg_color = "#F5F3E7"
    text_color = "#333333"
    sidebar_bg = "#F5F3E7"
    sidebar_text = "#333333"

if "last_action" in st.session_state and st.session_state["last_action"] not in ["upload_hewan", "upload_bunga"]:
    st.session_state["last_action"] = None

# ===== CSS dinamis =====
st.markdown(
    f"""
    <style>
        /* ===== Efek transisi lembut untuk semua elemen utama ===== */
        .stApp, .block-container, section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {{
            transition: background-color 0.4s ease, color 0.4s ease !important;
        }}

        /* ===== Tampilan utama ===== */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            color: {sidebar_text} !important;
        }}
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {sidebar_text} !important;
        }}

        /* ===== Heading & teks utama ===== */
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {text_color} !important;
        }}

        /* ===== Tombol ===== */
        .stButton > button {{
            background-color: {'#444444' if theme == "🌙 Gelap" else ('#E6E1C5' if theme == "📖 Redup / Baca" else '#F0F2F6')} !important;
            color: {text_color} !important;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            transition: background-color 0.3s ease, color 0.3s ease !important;
        }}
        .stButton > button:hover {{
            filter: brightness(1.1);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📂 Riwayat Prediksi Akan Ditampilkan di Bawah.**")

# ==========================
# Riwayat Prediksi
# ==========================
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None

# ==========================
# Header + Menu Utama di Atas
# ==========================
st.title("🌸🐾 Aplikasi Deteksi Bunga & Klasifikasi Hewan")

# --- Inisialisasi state ---
if "mode" not in st.session_state:
    st.session_state["mode"] = None  # Belum memilih mode

# --- Tombol navigasi ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🐾 Klasifikasi Hewan (CNN)", use_container_width=True):
        st.session_state["mode"] = "hewan"
with col2:
    if st.button("🌼 Deteksi Bunga (YOLO)", use_container_width=True):
        st.session_state["mode"] = "bunga"

# ==========================
# Mode Klasifikasi Hewan
# ==========================
if st.session_state["mode"] == "hewan":
    st.subheader("📘 Mode: Klasifikasi Hewan (TFLite)")
    st.info("Model ini akan mengklasifikasikan gambar menjadi **Kucing**, **Anjing**, atau **Satwa Liar**.")

    confirm = st.radio("Apakah kamu yakin gambar yang ingin diunggah adalah **hewan**?", ["Ya", "Tidak"])

    if confirm == "Tidak":
        st.warning("⚠️ Pastikan ulang bahwa gambar kamu adalah hewan sebelum mengunggah!")
    else:
        uploaded_file = st.file_uploader("Unggah gambar hewan di sini 🐾", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

            with st.spinner("🔍 Sedang menganalisis gambar..."):
                import time
                time.sleep(1.5)

                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

                input_details = tflite_interpreter.get_input_details()
                output_details = tflite_interpreter.get_output_details()

                tflite_interpreter.set_tensor(input_details[0]['index'], img_array)
                tflite_interpreter.invoke()
                prediction = tflite_interpreter.get_tensor(output_details[0]['index'])[0]

                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

            if confidence < 0.7:
                st.error("🚫 Gambar ini tidak dikenali sebagai hewan.")
            else:
                st.success(f"🐾 Hasil: **{animal_classes[class_index]}** ({confidence:.2%})")

                explanations = {
                    "Kucing": "Kucing adalah mamalia karnivora yang sering dipelihara manusia karena sifatnya yang lucu dan jinak.",
                    "Anjing": "Anjing adalah hewan sosial yang dikenal sebagai sahabat manusia dan sering dilatih untuk berbagai tugas.",
                    "Satwa Liar": "Satwa liar adalah hewan yang hidup di alam bebas, seperti harimau, singa, atau rubah."
                }
                st.info(f"📘 Penjelasan: {explanations[animal_classes[class_index]]}")

                st.session_state["history"].append({
                    "Model": "CNN (TFLite)",
                    "Prediksi": animal_classes[class_index],
                    "Akurasi": f"{confidence:.2%}"
                })
                st.session_state["last_action"] = "upload_hewan"

# ==========================
# Mode Deteksi Bunga
# ==========================
elif st.session_state["mode"] == "bunga":
    st.subheader("🌼 Mode: Deteksi Bunga (YOLO)")
    st.info("Model ini akan mendeteksi bunga **Daisy** dan **Dandelion**.")

    confirm = st.radio("Apakah kamu yakin gambar yang ingin diunggah adalah **bunga**?", ["Ya", "Tidak"])

    if confirm == "Tidak":
        st.warning("⚠️ Pastikan ulang bahwa gambar kamu adalah bunga sebelum mengunggah!")
    else:
        uploaded_file = st.file_uploader("Unggah gambar bunga di sini 🌸", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

            with st.spinner("🔍 Sedang mendeteksi objek..."):
                import time
                time.sleep(1.5)
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
                st.image(result_img, caption="🌸 Hasil Deteksi Bunga", use_container_width=True)

                for label, conf in valid_detections:
                    st.success(f"🌼 Terdeteksi: **{label}** ({conf:.2%})")

                    explanations = {
                        "Daisy": "Daisy memiliki kelopak putih dengan tengah berwarna kuning. Melambangkan kemurnian dan kesederhanaan.",
                        "Dandelion": "Dandelion dikenal dengan kelopak kuning cerah dan biji berbulu putih yang mudah tertiup angin."
                    }
                    st.info(f"📘 Penjelasan: {explanations[label]}")

                    st.session_state["history"].append({
                        "Model": "YOLO",
                        "Prediksi": label,
                        "Akurasi": f"{conf:.2%}"
                    })
                    st.session_state["last_action"] = "upload_bunga"

# ==========================
# Riwayat Prediksi
# ==========================
if st.session_state["last_action"] in ["upload_hewan", "upload_bunga"]:
    st.markdown("---")
    st.subheader("📊 Riwayat Prediksi")
    st.table(st.session_state["history"])
