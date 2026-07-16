from __future__ import annotations

from collections import Counter
from datetime import datetime
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlsplit, urlunsplit

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


def get_bardi_settings() -> dict[str, str]:
    """Read local RTSP settings without failing when Cloud has no secrets file."""
    try:
        settings = st.secrets.get("bardi", {})
        return {
            "rtsp_url": str(settings.get("rtsp_url", "")).strip(),
            "username": str(settings.get("username", "")).strip(),
            "password": str(settings.get("password", "")),
        }
    except Exception:
        return {"rtsp_url": "", "username": "", "password": ""}


def build_rtsp_url(base_url: str, username: str = "", password: str = "") -> str:
    """Validate an RTSP URL and optionally inject escaped credentials."""
    parts = urlsplit(base_url)
    if parts.scheme.lower() not in {"rtsp", "rtsps"} or not parts.hostname:
        raise ValueError("URL kamera harus menggunakan format rtsp:// atau rtsps://")
    if not username:
        return base_url

    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    if parts.port:
        host += f":{parts.port}"
    credentials = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    return urlunsplit((parts.scheme, credentials + host, parts.path, parts.query, parts.fragment))


def safe_rtsp_label(base_url: str) -> str:
    """Return an RTSP endpoint without embedded credentials for display."""
    parts = urlsplit(base_url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def capture_rtsp_frame(rtsp_url: str, timeout_ms: int = 8_000) -> Image.Image:
    """Open a BARDI RTSP stream over TCP and return the freshest available frame."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    import cv2

    capture = cv2.VideoCapture()
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

    try:
        if not capture.open(rtsp_url, cv2.CAP_FFMPEG):
            raise ConnectionError("Stream RTSP tidak dapat dibuka")

        frame = None
        for _ in range(3):
            success, candidate = capture.read()
            if success and candidate is not None:
                frame = candidate
        if frame is None:
            raise ConnectionError("Kamera terhubung tetapi frame tidak dapat dibaca")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    finally:
        capture.release()


def run_inference(model, image_sources, confidence: float, iou: float, image_size: int):
    rows: list[dict] = []
    rendered: list[tuple[str, object]] = []

    for image_name, source in image_sources:
        image = source.convert("RGB") if isinstance(source, Image.Image) else Image.open(source).convert("RGB")
        result = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            verbose=False,
        )[0]
        rendered.append((image_name, result.plot()[:, :, ::-1].copy()))

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
                    "gambar": image_name,
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
        ["Unggah gambar", "Kamera langsung", "Kamera BARDI (RTSP/ODM)"],
        horizontal=True,
        help="Kamera BARDI memerlukan aplikasi lokal yang satu jaringan dengan kamera.",
    )
    if input_source == "Unggah gambar":
        files = st.file_uploader(
            "Pilih satu atau beberapa gambar",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Semua hasil deteksi pada kumpulan gambar akan digabungkan.",
        )
        image_sources = [(uploaded_file.name, uploaded_file) for uploaded_file in files]
    elif input_source == "Kamera langsung":
        camera_capture = st.camera_input(
            "Arahkan kamera ke maggot, lalu ambil foto",
            help="Izinkan akses kamera pada browser. Bisa digunakan dari kamera HP maupun webcam komputer.",
        )
        image_sources = (
            [(camera_capture.name, camera_capture)] if camera_capture is not None else []
        )
        st.caption(
            "📱 Di HP, buka dashboard melalui browser dan izinkan akses kamera. "
            "Gunakan koneksi HTTPS agar izin kamera didukung dengan baik."
        )
    else:
        bardi_settings = get_bardi_settings()
        base_rtsp_url = bardi_settings["rtsp_url"]
        image_sources = []
        if not base_rtsp_url:
            st.warning(
                "Kamera BARDI belum dikonfigurasi. Isi bagian `[bardi]` pada "
                "`.streamlit/secrets.toml`, lalu jalankan aplikasi secara lokal."
            )
        else:
            st.caption("Endpoint ODM/RTSP aktif")
            st.code(safe_rtsp_label(base_rtsp_url), language=None)
            rtsp_username = st.text_input(
                "Username RTSP",
                value=bardi_settings["username"],
                placeholder="Contoh: admin",
                help="Kosongkan jika kamera tidak memakai autentikasi RTSP.",
            )
            rtsp_password = st.text_input(
                "Password RTSP",
                value="",
                type="password",
                placeholder="Kosong = gunakan password dari secrets",
                help="Password dari secrets tidak dikirim ke form. Isian baru hanya tersimpan selama sesi dashboard.",
            )
            if st.button("📸 Ambil gambar dari BARDI", type="primary", use_container_width=True):
                st.session_state.pop("bardi_capture", None)
                try:
                    authenticated_url = build_rtsp_url(
                        base_rtsp_url,
                        rtsp_username.strip(),
                        rtsp_password or bardi_settings["password"],
                    )
                    with st.spinner("Menghubungkan ke kamera BARDI dan mengambil frame…"):
                        bardi_frame = capture_rtsp_frame(authenticated_url)
                    capture_name = f"bardi_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    st.session_state["bardi_capture"] = (capture_name, bardi_frame)
                    st.success("Frame kamera berhasil diambil dan siap dianalisis.")
                except Exception as exc:
                    st.error(
                        f"Gagal mengambil frame BARDI: {exc}. Pastikan kamera dan laptop "
                        "berada pada Wi-Fi yang sama serta RTSP aktif di ODM."
                    )
            if "bardi_capture" in st.session_state:
                image_sources = [st.session_state["bardi_capture"]]
            st.caption(
                "🔒 Kredensial tidak ditampilkan pada hasil analisis dan tidak disimpan ke GitHub."
            )
with right:
    st.subheader("Aturan keputusan")
    st.markdown(
        "🌱 **Bayi > ambang** → belum siap panen  \n"
        "✅ **Remaja > ambang** → siap panen  \n"
        "⚠️ **Dewasa > ambang** → melewati siap panen"
    )

if not image_sources:
    empty_message = {
        "Unggah gambar": "Unggah gambar untuk memulai deteksi.",
        "Kamera langsung": "Ambil foto dari kamera untuk memulai deteksi.",
        "Kamera BARDI (RTSP/ODM)": "Ambil satu frame dari kamera BARDI untuk memulai deteksi.",
    }[input_source]
    st.info(empty_message, icon="📷")
    st.stop()

if not DEFAULT_MODEL.exists():
    st.error(f"Model tidak ditemukan di {DEFAULT_MODEL}")
    st.stop()

try:
    model = load_model(str(DEFAULT_MODEL))
    validate_model_classes(model.names)
    with st.spinner(f"Menganalisis {len(image_sources)} gambar…"):
        detections, rendered_images = run_inference(
            model, image_sources, confidence, iou, image_size
        )
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
metric_columns[3].metric(
    "Total objek", str(len(detections)), f"{len(image_sources)} gambar", delta_color="off"
)

st.subheader("Komposisi fase")
for class_name in EXPECTED_CLASSES:
    percentage = max(0.0, min(100.0, percentages[class_name]))
    st.progress(
        percentage / 100,
        text=f"**{CLASS_LABELS[class_name]}** — {percentage:.1f}%",
    )
st.caption("Progress bar dibuat noninteraktif agar tidak mengganggu scroll pada layar sentuh.")

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
