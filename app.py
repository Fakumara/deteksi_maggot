from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "yolo_maggot" / "weights" / "best.pt"
EXPECTED_CLASSES = ("Bayi_Maggot", "Remaja_Maggot", "Dewasa_Maggot")
CLASS_LABELS = {
    "Bayi_Maggot": "Bayi Maggot",
    "Remaja_Maggot": "Remaja Maggot",
    "Dewasa_Maggot": "Dewasa Maggot",
}
CONCLUSIONS = {
    "Bayi_Maggot": ("Belum siap panen", "🌱"),
    "Remaja_Maggot": ("Siap panen", "✅"),
    "Dewasa_Maggot": ("Melewati siap panen", "⚠️"),
}


st.set_page_config(
    page_title="Maggot Harvest Vision",
    page_icon="🪰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f8f3; }
        [data-testid="stSidebar"] { background: #e9efe4; }
        .hero {
            padding: 1.7rem 2rem; border-radius: 22px;
            background: linear-gradient(120deg, #183c2c 0%, #316c48 65%, #86a64a 100%);
            color: white; margin-bottom: 1.25rem;
            box-shadow: 0 12px 30px rgba(25, 65, 45, .15);
        }
        .hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.25rem); }
        .hero p { margin: .45rem 0 0; color: #e9f2e1; font-size: 1.05rem; }
        .result-card {
            padding: 1.25rem 1.4rem; border-radius: 18px; background: white;
            border-left: 7px solid #5d873f; box-shadow: 0 7px 20px rgba(22, 48, 35, .08);
        }
        .result-card h2 { margin: .15rem 0; color: #173d2b; }
        .result-card p { margin: 0; color: #5c695f; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #dde5d8; padding: 1rem;
            border-radius: 16px; box-shadow: 0 5px 14px rgba(22, 48, 35, .05);
        }
        .small-note { color: #687469; font-size: .88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Memuat model YOLO11…")
def load_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def confidence_weighted_percentages(
    rows: Iterable[dict], classes: Iterable[str] = EXPECTED_CLASSES
) -> dict[str, float]:
    """Return class shares based on the sum of detection confidences."""
    class_names = tuple(classes)
    totals = {name: 0.0 for name in class_names}
    for row in rows:
        name = str(row["kelas"])
        if name in totals:
            totals[name] += float(row["confidence"])
    total_score = sum(totals.values())
    if total_score == 0:
        return {name: 0.0 for name in class_names}
    return {name: score / total_score * 100 for name, score in totals.items()}


def make_conclusion(percentages: dict[str, float], threshold: float) -> tuple[str, str, str | None]:
    """Map a single class above threshold to the requested harvest conclusion."""
    winners = [name for name, value in percentages.items() if value > threshold]
    if not winners:
        return "Belum dapat disimpulkan", "🔎", None
    winner = max(winners, key=percentages.get)
    label, icon = CONCLUSIONS[winner]
    return label, icon, winner


def validate_model_classes(names: dict | list) -> None:
    available = set(names.values() if isinstance(names, dict) else names)
    missing = set(EXPECTED_CLASSES) - available
    if missing:
        raise ValueError("Model tidak memiliki kelas: " + ", ".join(sorted(missing)))


def run_inference(model, files, confidence: float, iou: float, image_size: int):
    rows: list[dict] = []
    rendered: list[tuple[str, object]] = []

    for uploaded_file in files:
        image = Image.open(uploaded_file).convert("RGB")
        result = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            verbose=False,
        )[0]
        rendered.append((uploaded_file.name, result.plot()[:, :, ::-1].copy()))

        if result.boxes is None:
            continue
        for class_id, score, xyxy in zip(
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.xyxy.cpu().tolist(),
        ):
            x1, y1, x2, y2 = xyxy
            rows.append(
                {
                    "gambar": uploaded_file.name,
                    "kelas": result.names[int(class_id)],
                    "confidence": float(score),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )
    return rows, rendered


apply_style()
st.markdown(
    """
    <div class="hero">
      <h1>Maggot Harvest Vision</h1>
      <p>Analisis fase pertumbuhan dan kesiapan panen berbasis YOLO11.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pengaturan analisis")
    confidence = st.slider("Confidence minimum", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.70, 0.05)
    threshold = st.slider("Ambang kesimpulan", 50, 90, 60, 1)
    image_size = st.select_slider("Ukuran inferensi", [320, 480, 640, 800, 960], value=640)
    st.divider()
    st.caption("Model aktif")
    st.code(str(DEFAULT_MODEL.relative_to(APP_DIR)), language=None)
    st.markdown(
        "<p class='small-note'>Persentase dihitung dari total confidence tiap kelas, lalu dinormalisasi menjadi 100%.</p>",
        unsafe_allow_html=True,
    )

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("Sumber foto maggot")
    input_source = st.radio(
        "Pilih sumber gambar",
        ["Unggah gambar", "Kamera langsung"],
        horizontal=True,
        help="Kamera yang digunakan mengikuti perangkat yang membuka dashboard.",
    )
    if input_source == "Unggah gambar":
        files = st.file_uploader(
            "Pilih satu atau beberapa gambar",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Semua hasil deteksi pada kumpulan gambar akan digabungkan.",
        )
    else:
        camera_capture = st.camera_input(
            "Arahkan kamera ke maggot, lalu ambil foto",
            help="Izinkan akses kamera pada browser. Bisa digunakan dari kamera HP maupun webcam komputer.",
        )
        files = [camera_capture] if camera_capture is not None else []
        st.caption(
            "📱 Di HP, buka dashboard melalui browser dan izinkan akses kamera. "
            "Gunakan koneksi HTTPS agar izin kamera didukung dengan baik."
        )
with right:
    st.subheader("Aturan keputusan")
    st.markdown(
        "🌱 **Bayi > ambang** → belum siap panen  \n"
        "✅ **Remaja > ambang** → siap panen  \n"
        "⚠️ **Dewasa > ambang** → melewati siap panen"
    )

if not files:
    empty_message = (
        "Unggah gambar untuk memulai deteksi."
        if input_source == "Unggah gambar"
        else "Ambil foto dari kamera untuk memulai deteksi."
    )
    st.info(empty_message, icon="📷")
    st.stop()

if not DEFAULT_MODEL.exists():
    st.error(f"Model tidak ditemukan di {DEFAULT_MODEL}")
    st.stop()

try:
    model = load_model(str(DEFAULT_MODEL))
    validate_model_classes(model.names)
    with st.spinner(f"Menganalisis {len(files)} gambar…"):
        detections, rendered_images = run_inference(model, files, confidence, iou, image_size)
except Exception as exc:
    st.error(f"Analisis gagal: {exc}")
    st.stop()

percentages = confidence_weighted_percentages(detections)
conclusion, icon, winner = make_conclusion(percentages, float(threshold))
winner_text = (
    f"{CLASS_LABELS[winner]} mendominasi {percentages[winner]:.1f}%"
    if winner
    else f"Tidak ada kelas yang melebihi {threshold}%"
)
st.markdown(
    f"""
    <div class="result-card">
      <p>Kesimpulan analisis</p>
      <h2>{icon} {conclusion}</h2>
      <p>{winner_text} dari skor deteksi gabungan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
metric_columns = st.columns(4)
counts = Counter(row["kelas"] for row in detections)
for column, class_name in zip(metric_columns[:3], EXPECTED_CLASSES):
    column.metric(
        CLASS_LABELS[class_name],
        f"{percentages[class_name]:.1f}%",
        f"{counts[class_name]} deteksi",
        delta_color="off",
    )
metric_columns[3].metric("Total objek", str(len(detections)), f"{len(files)} gambar", delta_color="off")

chart_data = pd.DataFrame(
    {
        "Fase": [CLASS_LABELS[name] for name in EXPECTED_CLASSES],
        "Persentase": [percentages[name] for name in EXPECTED_CLASSES],
    }
).set_index("Fase")
st.subheader("Komposisi fase")
st.bar_chart(chart_data, y="Persentase", color="#5d873f")

image_tab, data_tab, method_tab = st.tabs(["Hasil visual", "Data deteksi", "Metode hitung"])
with image_tab:
    if rendered_images:
        image_columns = st.columns(2)
        for index, (name, image) in enumerate(rendered_images):
            image_columns[index % 2].image(image, caption=name, use_container_width=True)
with data_tab:
    if detections:
        table = pd.DataFrame(detections)
        table["confidence"] = (table["confidence"] * 100).round(2)
        table = table.rename(columns={"confidence": "confidence (%)"})
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Unduh hasil CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name="hasil_deteksi_maggot.csv",
            mime="text/csv",
        )
    else:
        st.warning("Tidak ada objek terdeteksi pada confidence minimum saat ini.")
with method_tab:
    st.markdown(
        """
        Skor setiap kelas adalah **jumlah confidence seluruh objek pada kelas tersebut**.
        Skor kemudian dibagi total skor ketiga kelas sehingga persentasenya berjumlah 100%.
        Kesimpulan hanya diberikan bila satu kelas **lebih besar** dari ambang (default 60%).

        `persentase kelas = Σ confidence kelas / Σ confidence semua kelas × 100%`
        """
    )
