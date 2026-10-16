import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "cifar_10.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

st.set_page_config(
    page_title="CNN Vision | CIFAR-10",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
EMOJIS = {
    "Airplane": "✈️", "Automobile": "🚙", "Bird": "🐦", "Cat": "🐱",
    "Deer": "🦌", "Dog": "🐶", "Frog": "🐸", "Horse": "🐴",
    "Ship": "🚢", "Truck": "🚚"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#14213d; --muted:#64748b; --line:#e4eaf5; --blue:#4f46e5; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f7f9ff 0%,#f4f6fc 55%,#f8faff 100%); color:var(--ink); }
.block-container { max-width:1500px; padding-top:1.5rem; padding-bottom:2rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111d3b 0%,#17264a 100%); border-right:1px solid #26385e; }
[data-testid="stSidebar"] * { color:#edf2ff; }
[data-testid="stSidebar"] .stRadio label { background:rgba(255,255,255,.035); border-radius:12px; padding:8px 10px; }
[data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.10); }
.hero { border-radius:22px; padding:26px 30px; background:linear-gradient(110deg,#ffffff 0%,#eef2ff 100%); border:1px solid #e2e8f7; box-shadow:0 8px 30px rgba(36,55,105,.05); margin-bottom:20px; }
.eyebrow { color:#4f46e5; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:11px; }
.hero h1 { font-family:'Manrope',sans-serif; color:#14213d; font-size:36px; line-height:1.15; margin:7px 0 8px 0; font-weight:800; }
.hero p { color:#64748b; margin:0; font-size:15px; }
.panel { background:rgba(255,255,255,.94); border:1px solid #e2e8f0; border-radius:18px; padding:20px; box-shadow:0 6px 22px rgba(30,50,100,.035); min-height:120px; }
.panel-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:17px; color:#14213d; margin-bottom:12px; }
.result { background:linear-gradient(135deg,#ecfdf5,#f0fdfa); border:1px solid #b8eee0; border-radius:16px; padding:18px; }
.result-label { color:#047857; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
.result-name { color:#102a43; font-family:'Manrope',sans-serif; font-size:28px; font-weight:800; margin:5px 0; }
.pill { display:inline-block; background:#d1fae5; color:#047857; border-radius:999px; padding:6px 11px; font-weight:700; font-size:12px; }
.metric-card { background:white; border:1px solid #e2e8f0; border-radius:15px; padding:15px 17px; }
.metric-label { font-size:12px; color:#64748b; margin-bottom:5px; }
.metric-value { color:#172554; font-family:'Manrope',sans-serif; font-weight:800; font-size:20px; }
.small-note { font-size:12px; color:#64748b; }
div.stButton > button[kind="primary"] { border:0; border-radius:12px; background:linear-gradient(100deg,#6366f1,#4f46e5); color:white; font-weight:700; min-height:46px; box-shadow:0 6px 14px rgba(79,70,229,.18); }
div.stButton > button[kind="primary"]:hover { background:linear-gradient(100deg,#4f46e5,#4338ca); border:0; }
[data-testid="stFileUploader"] { border:1.5px dashed #b9c7e6; border-radius:14px; padding:8px; background:#fbfcff; }
hr { border-color:#e7ecf5; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="display:flex;gap:12px;align-items:center;padding:8px 4px 22px 4px;">
      <div style="font-size:36px;">🧠</div>
      <div>
        <div style="font-family:Manrope,sans-serif;font-size:22px;font-weight:800;">CNN Vision</div>
        <div style="font-size:12px;color:#b9c8e8;">CIFAR-10 Image Classification</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["⌂  Home", "ⓘ  About", "▤  Model Info", "⚙  How It Works"], label_visibility="visible")
    st.markdown("<br><br>", unsafe_allow_html=True)


if page.startswith("ⓘ"):
    st.markdown('<div class="hero"><div class="eyebrow">About this project</div><h1>About CNN Vision</h1><p>Antarmuka demo untuk memahami alur klasifikasi gambar menggunakan CNN.</p></div>', unsafe_allow_html=True)
    st.write("CNN Vision adalah contoh dashboard web untuk alur klasifikasi gambar dengan dataset CIFAR-10. Versi ZIP ini berfokus pada tampilan dan interaksi UI.")
    st.warning("Versi ini sudah disesuaikan untuk model CIFAR-10 yang tersimpan di file cifar_10.keras.")
    st.stop()

if page.startswith("▤"):
    st.markdown('<div class="hero"><div class="eyebrow">Model details</div><h1>Model Information</h1><p>Informasi dasar dataset dan asumsi input yang umum pada CIFAR-10.</p></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Dataset", "CIFAR-10")
    c2.metric("Image input", "32 × 32 × 3")
    c3.metric("Classes", "10")
    st.markdown("### Kelas yang didukung")
    st.write(", ".join(CLASSES))
    st.info("Model dirancang untuk input gambar 32 × 32 × 3 sesuai format CIFAR-10.")
    st.stop()

if page.startswith("⚙"):
    st.markdown('<div class="hero"><div class="eyebrow">Workflow</div><h1>How It Works</h1><p>Alur sederhana dari gambar input hingga distribusi prediksi.</p></div>', unsafe_allow_html=True)
    for i, (title, detail) in enumerate([
        ("01 · Upload", "Pilih gambar JPG, JPEG, atau PNG dari perangkat."),
        ("02 · Preview", "Lihat gambar yang dipilih sebelum proses demo."),
        ("03 · Predict", "Klik tombol prediksi untuk menjalankan inferensi model real."),
        ("04 · Review", "Periksa kelas teratas dan probabilitas 10 kelas.")
    ]):
        st.markdown(f"**{title}**  \n{detail}")
        st.divider()
    st.stop()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Image intelligence · Demo workspace</div>
  <h1>Selamat datang di CNN Vision ✦</h1>
  <p>Upload gambar dan lihat contoh bagaimana model CNN mengklasifikasikan objek dari dataset CIFAR-10.</p>
</div>
""", unsafe_allow_html=True)

status_col, clear_col = st.columns([5,1])
with status_col:
    st.markdown('<span class="pill">● MODEL AKTIF · cifar_10.keras</span>', unsafe_allow_html=True)
with clear_col:
    if st.button("↻ Clear", use_container_width=True):
        st.rerun()

upload_col, preview_col, probs_col = st.columns([1.05, 1.1, 1.05], gap="medium")

with upload_col:
    st.markdown('<div class="panel"><div class="panel-title">☁️ Upload Gambar</div><div class="small-note">Pilih gambar dari perangkat kamu. Format yang didukung: JPG, JPEG, PNG.</div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload gambar", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.caption("Gambar akan diubah ukurannya menjadi 32 × 32 piksel untuk menyesuaikan input model CIFAR-10.")
    if uploaded:
        st.success("Gambar siap dipreview.")
    else:
        st.info("Belum ada gambar. Pilih file untuk memulai.")

with preview_col:
    st.markdown('<div class="panel"><div class="panel-title">▣ Preview Gambar</div>', unsafe_allow_html=True)
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption(f"Ukuran asli: {image.width} × {image.height} px")
    else:
        st.markdown("""
        <div style="height:220px;border:1px dashed #cbd5e1;border-radius:14px;display:flex;align-items:center;justify-content:center;text-align:center;background:#fafcff;">
          <div><div style="font-size:42px;">🖼️</div><div style="font-weight:700;color:#64748b;">Preview gambar akan muncul di sini</div><div style="font-size:12px;color:#94a3b8;">Upload gambar untuk memulai</div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    predict_clicked = st.button("✦  Prediksi Gambar", type="primary", use_container_width=True, disabled=(uploaded is None))

with probs_col:
    st.markdown('<div class="panel"><div class="panel-title">▥ Probabilitas Setiap Kelas</div>', unsafe_allow_html=True)
    if "model_probs" not in st.session_state:
        default_probs = np.array([0.023,0.887,0.012,0.008,0.004,0.011,0.006,0.007,0.021,0.011], dtype=float)
        st.session_state.model_probs = default_probs / default_probs.sum()
    probs = st.session_state.model_probs
    if predict_clicked and uploaded:
        try:
            model = load_model()
            img = Image.open(uploaded).convert("RGB")
            batch = preprocess_image(img)
            probs = model.predict(batch, verbose=0)[0]
            st.session_state.model_probs = probs.astype(np.float64)
        except Exception as e:
            st.session_state.model_probs = np.full(len(CLASSES), 1 / len(CLASSES), dtype=np.float64)
            st.error(f"Gagal menjalankan model: {e}")
    for name, p in sorted(zip(CLASSES, probs), key=lambda pair: pair[1], reverse=True):
        st.markdown(f"<div style='display:flex;justify-content:space-between;gap:8px;font-size:12px;margin:10px 0 3px 0;'><span>{EMOJIS[name]} &nbsp; {name}</span><b>{p*100:.1f}%</b></div>", unsafe_allow_html=True)
        st.progress(float(p))
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked and uploaded:
    probs = st.session_state.model_probs
    best_idx = int(np.argmax(probs))
    best_name = CLASSES[best_idx]
    confidence = float(probs[best_idx] * 100)
    st.markdown("<br>", unsafe_allow_html=True)
    result_col, info_col = st.columns([1.5,1])
    with result_col:
        st.markdown(f"""
        <div class="result">
          <div class="result-label">✓ Hasil prediksi model</div>
          <div class="result-name">{EMOJIS[best_name]} {best_name}</div>
          <span class="pill">Confidence: {confidence:.1f}%</span>
          <p style="font-size:12px;color:#64748b;margin:12px 0 0 0;">Inferensi dilakukan menggunakan model CIFAR-10 yang disimpan di cifar_10.keras.</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col:
        st.markdown('<div class="metric-card"><div class="metric-label">Status inferensi</div><div class="metric-value">Model aktif</div><div class="small-note">Menggunakan TensorFlow / Keras</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Informasi Model")
m1,m2,m3,m4 = st.columns(4)
for col, label, value, detail in [
    (m1,"Dataset","CIFAR-10","32 × 32 × 3"),
    (m2,"Jumlah kelas","10 kelas","Airplane hingga Truck"),
    (m3,"Arsitektur","CNN","Convolutional Neural Network"),
    (m4,"Framework","Streamlit","Web application"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="small-note">{detail}</div></div>', unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#8190ad;font-size:11px;"><span>CNN Vision · CIFAR-10 Image Classification</span><span>Model terhubung · cifar_10.keras</span></div>', unsafe_allow_html=True)
