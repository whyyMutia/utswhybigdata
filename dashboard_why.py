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
animal_classes_en = ["Cat", "Dog", "Wild Animal"]

flower_classes = ["Daisy", "Dandelion"]

# ==========================
# Sidebar – Pengaturan
# ==========================
st.sidebar.header("🛠️ Pengaturan")

# Tema
theme = st.sidebar.radio("Pilih Mode Tampilan:", ["🌞 Terang", "🌙 Gelap", "📖 Redup / Baca"])

# Ukuran gambar
img_size = st.sidebar.selectbox("Ukuran Gambar:", ["Kecil", "Sedang", "Besar"])
size_dict = {"Kecil": 128, "Sedang": 224, "Besar": 384}
img_display_size = size_dict[img_size]

# Bahasa
language = st.sidebar.radio("Bahasa Tampilan:", ["Indonesia", "English"])

# Panduan Pengguna
with st.sidebar.expander("📖 Panduan Pengguna"):
    if language == "Indonesia":
        st.markdown("""
        - Pilih mode: **Klasifikasi Hewan** atau **Deteksi Bunga**  
        - Unggah gambar sesuai mode yang dipilih  
        - Tunggu hingga prediksi muncul  
        - Lihat riwayat prediksi di bawah
        """)
    else:
        st.markdown("""
        - Select mode: **Animal Classification** or **Flower Detection**  
        - Upload image according to selected mode  
        - Wait for prediction results  
        - Check prediction history below
        """)

# ==========================
# Terapkan tema
# ==========================
if theme == "🌞 Terang":
    bg_color = "#FFFFFF"; text_color = "#000000"; sidebar_bg = "#F8F9FA"; sidebar_text = "#000000"
elif theme == "🌙 Gelap":
    bg_color = "#0E1117"; text_color = "#FAFAFA"; sidebar_bg = "#0E1117"; sidebar_text = "#FAFAFA"
elif theme == "📖 Redup / Baca":
    bg_color = "#F5F3E7"; text_color = "#333333"; sidebar_bg = "#F5F3E7"; sidebar_text = "#333333"

st.markdown(
    f"""
    <style>
    .stApp, .block-container, section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {{
        transition: background-color 0.4s ease, color 0.4s ease !important;
    }}
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; color: {sidebar_text} !important; }}
    section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{ color: {sidebar_text} !important; }}
    h1, h2, h3, h4, h5, h6, p, label, span {{ color: {text_color} !important; }}
    .stButton > button {{
        background-color: {'#444444' if theme == "🌙 Gelap" else ('#E6E1C5' if theme == "📖 Redup / Baca" else '#F0F2F6')} !important;
        color: {text_color} !important;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1em;
        transition: background-color 0.3s ease, color 0.3s ease !important;
    }}
    .stButton > button:hover {{ filter: brightness(1.1); }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Inisialisasi session state
# ==========================
if "history" not in st.session_state: st.session_state["history"] = []
if "last_action" not in st.session_state: st.session_state["last_action"] = None
if "mode" not in st.session_state: st.session_state["mode"] = None

# ==========================
# Dictionary teks sesuai bahasa
# ==========================
texts = {
    "id": {
        "title": "🌸🐾 Aplikasi Deteksi Bunga & Klasifikasi Hewan",
        "mode_hewan": "📘 Mode: Klasifikasi Hewan (TFLite)",
        "mode_bunga": "🌼 Mode: Deteksi Bunga (YOLO)",
        "info_hewan": "Model ini akan mengklasifikasikan gambar menjadi **Kucing**, **Anjing**, atau **Satwa Liar**.",
        "info_bunga": "Model ini akan mendeteksi bunga **Daisy** dan **Dandelion**.",
        "confirm_hewan": "Apakah kamu yakin gambar yang ingin diunggah adalah **hewan**?",
        "confirm_bunga": "Apakah kamu yakin gambar yang ingin diunggah adalah **bunga**?",
        "warning": "⚠️ Pastikan ulang bahwa gambar kamu sesuai sebelum mengunggah!",
        "error_no_animal": "🚫 Gambar ini tidak dikenali sebagai hewan.",
        "error_no_flower": "🚫 Tidak terdeteksi bunga Daisy atau Dandelion. Pastikan gambar sesuai.",
        "history": "📊 Riwayat Prediksi"
    },
    "en": {
        "title": "🌸🐾 Flower Detection & Animal Classification App",
        "mode_hewan": "📘 Mode: Animal Classification (TFLite)",
        "mode_bunga": "🌼 Mode: Flower Detection (YOLO)",
        "info_hewan": "This model classifies the image into **Cat**, **Dog**, or **Wild Animal**.",
        "info_bunga": "This model detects **Daisy** and **Dandelion** flowers.",
        "confirm_hewan": "Are you sure the uploaded image is an **animal**?",
        "confirm_bunga": "Are you sure the uploaded image is a **flower**?",
        "warning": "⚠️ Make sure the image is correct before uploading!",
        "error_no_animal": "🚫 This image is not recognized as an animal.",
        "error_no_flower": "🚫 No Daisy or Dandelion detected. Please upload a valid flower image.",
        "history": "📊 Prediction History"
    }
}
lang = "id" if language == "Indonesia" else "en"

# ==========================
# Header + Menu Utama
# ==========================
st.title(texts[lang]["title"])
menu_texts = {
    "id": {
        "animal_button": "🐾 Klasifikasi Hewan (CNN)",
        "flower_button": "🌼 Deteksi Bunga (YOLO)"
    },
    "en": {
        "animal_button": "🐾 Animal Classification (CNN)",
        "flower_button": "🌼 Flower Detection (YOLO)"
    }
}
col1, col2 = st.columns(2)
with col1:
    if st.button(menu_texts[lang]["animal_button"], use_container_width=True):
        st.session_state["mode"] = "hewan"
with col2:
    if st.button(menu_texts[lang]["flower_button"], use_container_width=True):
        st.session_state["mode"] = "bunga"

# ==========================
# Mode Klasifikasi Hewan
# ==========================
if st.session_state["mode"] == "hewan":
    st.subheader(texts[lang]["mode_hewan"])
    st.info(texts[lang]["info_hewan"])

    confirm = st.radio(texts[lang]["confirm_hewan"], ["Ya", "Tidak"] if language=="Indonesia" else ["Yes", "No"])
    if confirm == ("Tidak" if language=="Indonesia" else "No"):
        st.warning(texts[lang]["warning"])
    else:
        uploaded_file = st.file_uploader("Unggah gambar hewan di sini 🐾" if language=="Indonesia" else "Upload animal image 🐾", type=["jpg","jpeg","png"])
        if uploaded_file:
            if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
                st.session_state["last_uploaded"] = uploaded_file.name
                is_new_upload = True
            else:
                is_new_upload = False

           # Buka gambar
            img = Image.open(uploaded_file).convert("RGB")

            # --- Resize untuk display sesuai pilihan user ---
            img_display = img.resize((img_display_size, img_display_size))
            st.image(img_display, caption="📸 Gambar yang Diupload", width=img_display_size)

            with st.spinner("🔍 Sedang menganalisis gambar..." if language=="Indonesia" else "🔍 Analyzing image..."):
                import time; time.sleep(1.5)

                # --- Resize untuk model TFLite sesuai input model ---
                input_details = tflite_interpreter.get_input_details()
                output_details = tflite_interpreter.get_output_details()
                input_shape = input_details[0]['shape'][1:3]  # ex: (224,224)
                img_for_model = img.resize((input_shape[1], input_shape[0]))
                img_array = np.expand_dims(np.array(img_for_model)/255.0, axis=0).astype(np.float32)

                # Prediksi
                tflite_interpreter.set_tensor(input_details[0]['index'], img_array)
                tflite_interpreter.invoke()
                prediction = tflite_interpreter.get_tensor(output_details[0]['index'])[0]

                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

            if confidence < 0.5:
                st.error(texts[lang]["error_no_animal"])
            else:
                label = animal_classes[class_index] if language=="Indonesia" else animal_classes_en[class_index]
                st.success(f"🐾 Hasil: **{label}** ({confidence:.2%})")

                animal_explanations = {
                    "id": {
                        "Kucing": "Kucing adalah mamalia karnivora yang sering dipelihara manusia karena sifatnya yang lucu dan jinak.",
                        "Anjing": "Anjing adalah hewan sosial yang dikenal sebagai sahabat manusia dan sering dilatih untuk berbagai tugas.",
                        "Satwa Liar": "Satwa liar adalah hewan yang hidup di alam bebas, seperti harimau, singa, atau rubah.",
                    },
                    "en": {
                        "Cat": "Cats are carnivorous mammals commonly kept as pets due to their cute and friendly nature.",
                        "Dog": "Dogs are social animals known as human companions and trained for various tasks.",
                        "Wild Animal": "Wild animals live freely in nature, like tigers, lions, or foxes."
                    }
                }
                st.info(f"📘 Penjelasan: {animal_explanations[lang][label]}")

                if is_new_upload:
                    st.session_state["history"].append({
                        "Model": "CNN (TFLite)",
                        "Prediksi": label,
                        "Akurasi": f"{confidence:.2%}"
                    })
                    st.session_state["last_action"] = "upload_hewan"

# ==========================
# Mode Deteksi Bunga
# ==========================
elif st.session_state["mode"] == "bunga":
    st.subheader(texts[lang]["mode_bunga"])
    st.info(texts[lang]["info_bunga"])

    confirm = st.radio(texts[lang]["confirm_bunga"], ["Ya", "Tidak"] if language=="Indonesia" else ["Yes", "No"])
    if confirm == ("Tidak" if language=="Indonesia" else "No"):
        st.warning(texts[lang]["warning"])
    else:
        uploaded_file = st.file_uploader("Unggah gambar bunga di sini 🌸" if language=="Indonesia" else "Upload flower image 🌸", type=["jpg","jpeg","png"])
        if uploaded_file:
            if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
                st.session_state["last_uploaded"] = uploaded_file.name
                is_new_upload = True
            else:
                is_new_upload = False

            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Gambar yang Diupload", width=img_display_size)

            with st.spinner("🔍 Sedang mendeteksi objek..." if language=="Indonesia" else "🔍 Detecting objects..."):
                import time; time.sleep(1.5)
                results = yolo_model(img)
                boxes = results[0].boxes

            valid_detections = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_idx in [0,1] and conf >= 0.6:
                        valid_detections.append((flower_classes[cls_idx], conf))

            if not valid_detections:
                st.error(texts[lang]["error_no_flower"])
            else:
                #tampilkan gambar dengan bounding box semua bunga
                result_img = results[0].plot()
                result_img = Image.fromarray(result_img).resize((img_display_size, img_display_size))
                st.image(result_img, caption="🌸 Hasil Deteksi Bunga", width=img_display_size)

                #ambil confidence maksimum tiap jenis bunga supaya deteksi hanya muncul sekali
                flower_max_conf = {}
                for label, conf in valid_detections:
                    if label not in flower_max_conf of conf > flower_max_conf[label]:
                        flower_max_conf[label] = conf

                #label unik untuk penjelasan (hanya muncul sekali per jenis bunga)
                flower_explanations = {
                    "id": {
                        "Daisy": "Daisy memiliki kelopak putih dengan tengah berwarna kuning. Melambangkan kemurnian dan kesederhanaan.",
                        "Dandelion": "Dandelion dikenal dengan kelopak kuning cerah dan biji berbulu putih yang mudah tertiup angin."
                    },
                    "en": {
                        "Daisy": "Daisy has white petals with a yellow center. It symbolizes purity and simplicity.",
                        "Dandelion": "Dandelion is known for bright yellow petals and white fluffy seeds easily blown by the wind."
                    }
                }
                for label, conf in flower_max_conf.items():
                    st.success(f"🌼 Terdeteksi: **{label}** ({conf:.2%})")
                    st.info(f"📘 Penjelasan: {flower_explanations[lang][label]}")

                    if is_new_upload:
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
    st.subheader(texts[lang]["history"])
    st.table(st.session_state["history"])
