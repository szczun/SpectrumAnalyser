import io
import time
import streamlit as st
import numpy as np
import librosa
from scipy.signal import get_window
import plotly.graph_objects as go
from streamlit import session_state as ss
from fft_analyzer import setup_fft_analysis, analyze_frame_fft
from fir_analyzer import setup_fir_analysis, analyze_frame_fir

st.set_page_config(page_title="Analyzer", layout="wide")
st.title("🎵 Spectrum Analyzer")

for key, default in {
    "analysis_running": False,
    "current_frame": 0,
    "analysis_data": None,
    "data_key": None,
    "ema_vals": None,
}.items():
    if key not in ss:
        ss[key] = default

def start_analysis():
    if ss.analysis_data is not None:
        ss.analysis_running = True
        ss.current_frame = 0
        ss.ema_vals = np.zeros(len(ss.analysis_data["bands"]), dtype=np.float32)

def stop_analysis():
    ss.analysis_running = False

@st.cache_data(show_spinner=False)
def prepare_data(file_bytes, method):
    y, fs = librosa.load(io.BytesIO(file_bytes), sr=44100, mono=True)
    y = y.astype(np.float32)

    N, hop = 4096, 1024
    if len(y) < N:
        y = np.pad(y, (0, N - len(y)))

    frames = []
    for start in range(0, len(y) - N + 1, hop):
        frames.append(y[start:start + N])
    frames = np.stack(frames, axis=0)

    def get_bands(fmin=63, fmax=8000, f_ref=1000.0):
        bands = []
        b = 3
        k = int(np.ceil(b * np.log2(fmin / f_ref)))
        while True:
            fc = f_ref * (2.0 ** (k / b))
            if fc > fmax:
                break
            f1 = fc / (2 ** (1 / 6))
            f2 = fc * (2 ** (1 / 6))
            if f2 >= fmin and f1 <= fmax:
                bands.append((fc, max(f1, fmin), min(f2, fmax)))
            k += 1
        return bands

    bands = get_bands()
    band_names = [f"{int(round(fc))} Hz" for (fc, _, _) in bands]
    w = get_window("hamming", N, fftbins=True).astype(np.float32)

    if method == "FFT binning":
        freqs, idx_map = setup_fft_analysis(N, fs, bands)
        setup = {"type": "FFT", "freqs": freqs, "idx_map": idx_map, "color": "cornflowerblue"}
    else:
        fir_bank = setup_fir_analysis(fs, bands, numtaps=1011, window_type="hamming")
        setup = {"type": "FIR", "fir_bank": fir_bank, "color": "orange"}

    return {
        "audio": y,
        "fs": fs,
        "frames": frames,
        "bands": bands,
        "band_names": band_names,
        "window": w,
        "setup": setup,
        "N": N,
        "hop": hop,
        "duration": len(y) / fs,
        "window_type": "hamming",
    }

col1, col2, col3, col4 = st.columns(4)
with col1:
    method = st.selectbox("Method", ["FFT binning", "FIR bank"], disabled=ss.analysis_running)
with col2:
    speed = st.slider("Speed", 0.1, 5.0, 3.0, disabled=ss.analysis_running)
with col3:
    st.button("▶️ Start", on_click=start_analysis, disabled=ss.analysis_running, type="primary")
with col4:
    st.button("⏹️ Stop", on_click=stop_analysis, disabled=not ss.analysis_running)

uploaded = st.file_uploader("Upload audio", type=["wav", "flac", "ogg", "mp3"])

if not uploaded:
    st.info("Upload audio file first")
    st.stop()

file_bytes = uploaded.getvalue()
current_data_key = (uploaded.name, len(file_bytes), method)

if ss.data_key != current_data_key:
    ss.analysis_data = prepare_data(file_bytes, method)
    ss.data_key = current_data_key
    ss.analysis_running = False
    ss.current_frame = 0
    ss.ema_vals = np.zeros(len(ss.analysis_data["bands"]), dtype=np.float32)

data = ss.analysis_data

with st.sidebar:
    st.header("📊 Analysis Parameters")
    st.write(f"**Filename:** `{uploaded.name}`")
    st.write(f"**Duration:** {data['duration']:.2f} s")
    st.write(f"**File Size:** {len(file_bytes) / 1024:.1f} KB")
    st.write(f"**Sample Frequency:** {data['fs']} Hz")
    st.write(f"**Frame Length:** {data['N']}")
    st.write(f"**Hop Size:** {data['hop']}")
    st.write(f"**Window Type:** {data['window_type'].title()}")
    st.write(f"**Frequency Resolution:** {data['fs'] / data['N']:.2f} Hz/bin")
    st.write(f"**Method:** {method}")

chart_placeholder = st.empty()
progress_placeholder = st.empty()
status_placeholder = st.empty()

if ss.analysis_running and ss.current_frame < len(data["frames"]):
    frame = data["frames"][ss.current_frame]

    if data["setup"]["type"] == "FFT":
        powers = analyze_frame_fft(
            frame,
            data["window"],
            data["setup"]["idx_map"],
        )
    else:
        powers = analyze_frame_fir(frame, data["setup"]["fir_bank"])

    alpha = 0.8
    for i, p in enumerate(powers):
        ss.ema_vals[i] = alpha * ss.ema_vals[i] + (1.0 - alpha) * p

    fig = go.Figure(
        data=[go.Bar(
            x=data["band_names"],
            y=ss.ema_vals,
            marker_color=data["setup"]["color"]
        )]
    )
    fig.update_layout(
        title=f"{method}",
        yaxis_title="RMS Power",
        xaxis_tickangle=45,
        height=500,
        template="plotly",
        uirevision="fixed"
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True, key="spectrum_chart")
    progress_placeholder.progress((ss.current_frame + 1) / len(data["frames"]))
    status_placeholder.info("🔴 Analysis Running")

    ss.current_frame += 1

    if ss.current_frame >= len(data["frames"]):
        ss.analysis_running = False
        status_placeholder.success("✅ Complete!")
    else:
        time.sleep(max(0.02, 0.1 / speed))
        st.rerun()

else:
    status_placeholder.info("⏸️ Stopped")
    if ss.ema_vals is not None:
        fig = go.Figure(
            data=[go.Bar(x=data["band_names"], y=ss.ema_vals, marker_color="gray")]
        )
        fig.update_layout(title="Last Result", yaxis_title="RMS Power", height=500, template="plotly")
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="last_result_chart")
    else:
        chart_placeholder.info("Click Start to begin analysis")
