import os
import sys
import subprocess
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import cv2
import numpy as np
import time
import threading
import tempfile
from datetime import datetime
from tensorflow.keras.models import load_model

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(BASE_PATH, "fall_evidence")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardianAI - Fall Detection",
    page_icon="🚨",
    layout="wide"
)

# ── Voice Alert ───────────────────────────────────────────────────────────────
def play_alert():
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.call([
                "PowerShell", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 1; "
                "$s.Speak('Alert! A person has fallen down! Immediate assistance required!')"
            ])
        elif sys.platform == "darwin":
            import subprocess
            subprocess.call(["say", "-v", "Samantha",
                           "Alert! A person has fallen down!"])
        else:
            import subprocess
            subprocess.call(["espeak", "Alert! A person has fallen down!"])
    except Exception as e:
        print(f"Voice alert error: {e}")

def trigger_alert():
    t = threading.Thread(target=play_alert, daemon=True)
    t.start()

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detection_model():
    model_path = os.path.join(BASE_PATH, "fall_video_model.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = load_detection_model()

# ── Session State ─────────────────────────────────────────────────────────────
for key, val in [
    ('conf_history', [1.0] * 30),
    ('fall_count', 0),
    ('is_falling', False),
    ('last_alert_time', 0.0)
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

source = st.sidebar.radio(
    "📹 Input Source",
    ["🎥 Webcam", "📁 Upload Video", "📂 Video from Folder"],
    index=0
)

uploaded_file = None
folder_path   = None

if source == "📁 Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload any video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload fall or normal video to test"
    )

elif source == "📂 Video from Folder":
    folder_path = st.sidebar.text_input(
        "Video path",
        value="videos\\fall\\Raw_Video\\20240912_102330.mp4"
    )
    st.sidebar.markdown("**Quick Select:**")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("⚠️ Fall Video"):
        st.session_state.sel = "videos\\fall\\Raw_Video\\20240912_102330.mp4"
    if c2.button("✅ Normal Video"):
        st.session_state.sel = "videos\\No_Fall\\Raw_Video\\C_N_450.mp4"
    if 'sel' in st.session_state:
        folder_path = st.session_state.sel

st.sidebar.markdown("---")
THRESHOLD      = st.sidebar.slider(
    "🎯 Fall Sensitivity Threshold", 0.10, 0.99, 0.95,
    help="Score below this = FALL. Keep at 0.95"
)
CONFIRM_FRAMES = st.sidebar.slider(
    "✅ Confirmation Frames", 1, 10, 2,
    help="Consecutive fall frames to trigger alert"
)
ALERT_COOLDOWN = st.sidebar.slider(
    "🔔 Alert Cooldown (sec)", 3, 30, 8,
    help="Minimum seconds between voice alerts"
)
SPEED = st.sidebar.slider(
    "⏩ Speed Delay", 0.0, 0.1, 0.01,
    help="0 = fastest"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Score Guide:**")
st.sidebar.markdown("🔴 Score → 0.0 = **FALL**")
st.sidebar.markdown("🟢 Score → 1.0 = **Normal**")
st.sidebar.info(f"📸 Evidence saved in: `fall_evidence/`")

IMG_SIZE        = 64
SEQUENCE_LENGTH = 10

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚨 GuardianAI: Fall Detection Dashboard")
st.markdown("Supports **Webcam**, **Uploaded Videos** and **Local Video Files**")
st.markdown("---")

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    run      = st.checkbox("▶️ Start Detection", value=False)
    st_frame = st.empty()
    info_box = st.empty()

with col2:
    st.subheader("📊 Live Confidence Score")
    st.caption("0 = Fall 🔴  |  1 = Normal 🟢")
    graph_area = st.empty()
    st.markdown("---")
    m1, m2 = st.columns(2)
    fall_metric   = m1.empty()
    status_metric = m2.empty()
    st.markdown("---")

    if st.button("🔄 Reset Alerts"):
        st.session_state.fall_count      = 0
        st.session_state.is_falling      = False
        st.session_state.conf_history    = [1.0] * 30
        st.session_state.last_alert_time = 0.0
        if 'sel' in st.session_state:
            del st.session_state['sel']
        st.rerun()

    if st.button("🔊 Test Voice Alert"):
        trigger_alert()
        st.success("Voice alert triggered!")

# ── Model check ───────────────────────────────────────────────────────────────
if not model:
    st.error("❌ `fall_video_model.h5` not found! Place it in the Fall folder.")
    st.stop()

# ── Instructions when not running ─────────────────────────────────────────────
if not run:
    if source == "🎥 Webcam":
        st.info("👆 Click **Start Detection** to open webcam")
        st.markdown("""
        **How to test:**
        - Stand/sit in front of camera → should show **Normal** 🟢
        - Lie down flat → should show **FALL DETECTED** 🔴
        - Voice alert will play automatically on fall
        """)
    elif source == "📁 Upload Video":
        if not uploaded_file:
            st.info("👆 Upload a video from the sidebar, then click **Start Detection**")
        else:
            st.success(f"✅ Ready: `{uploaded_file.name}` — click **Start Detection**")
    else:
        st.info("👆 Enter video path or use Quick Select, then click **Start Detection**")
    st.stop()

# ── Open Video Source ─────────────────────────────────────────────────────────
tmp_file = None

if source == "🎥 Webcam":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    source_name = "WEBCAM"
    info_box.info("📷 Webcam active — lie down to test fall detection")

elif source == "📁 Upload Video":
    if not uploaded_file:
        st.warning("⚠️ Please upload a video file first!")
        st.stop()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp_file.write(uploaded_file.read())
    tmp_file.close()
    cap = cv2.VideoCapture(tmp_file.name)
    source_name = uploaded_file.name
    info_box.info(f"🎬 Testing: `{uploaded_file.name}`")

else:
    if not folder_path or not os.path.exists(folder_path):
        st.error(f"❌ File not found: `{folder_path}`")
        st.markdown("**Available videos:**")
        for root, dirs, files in os.walk("videos"):
            for f in files:
                if f.endswith('.mp4'):
                    st.code(os.path.join(root, f))
        st.stop()
    cap = cv2.VideoCapture(folder_path)
    source_name = os.path.basename(folder_path)
    info_box.info(f"🎬 Testing: `{source_name}`")

if not cap.isOpened():
    st.error("❌ Cannot open video source!")
    st.stop()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = st.progress(0) if source != "🎥 Webcam" else None

# ── Detection Loop ────────────────────────────────────────────────────────────
frames_buffer = []
fall_counter  = 0
frame_count   = 0
last_pred     = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        if source != "🎥 Webcam":
            info_box.success(
                f"✅ Done! {frame_count} frames processed. "
                f"Falls detected: {st.session_state.fall_count}"
            )
        break

    frame_count += 1

    if source == "🎥 Webcam":
        frame = cv2.flip(frame, 1)

    # Preprocess
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    frames_buffer.append(resized)
    if len(frames_buffer) > SEQUENCE_LENGTH:
        frames_buffer.pop(0)

    # Predict every 5 frames
    if len(frames_buffer) == SEQUENCE_LENGTH and frame_count % 5 == 0:
        inp       = np.array(frames_buffer).reshape(
            1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3
        )
        last_pred = float(model.predict(inp, verbose=0)[0][0])

        # Update graph
        st.session_state.conf_history.append(last_pred)
        st.session_state.conf_history.pop(0)
        graph_area.line_chart(st.session_state.conf_history)

        # Detection logic
        if last_pred < THRESHOLD:
            fall_counter += 1
        else:
            fall_counter = 0
            st.session_state.is_falling = False

        # Trigger alert
        if fall_counter >= CONFIRM_FRAMES:
            now = time.time()
            if not st.session_state.is_falling:
                st.session_state.fall_count += 1

                # Save snapshot
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(LOG_DIR, f"fall_{ts}.jpg")
                cv2.imwrite(fname, frame)
                st.toast(
                    f"🚨 Fall #{st.session_state.fall_count} detected & saved!",
                    icon="🚨"
                )
                st.session_state.is_falling = True

            # Voice alert with cooldown
            if now - st.session_state.last_alert_time > ALERT_COOLDOWN:
    # Force run alert directly (not in thread)
             
             subprocess.Popen([
        "PowerShell", "-Command",
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 1; "
        "$s.Speak('Alert! A person has fallen down! Immediate assistance required!')"
    ])
    st.session_state.last_alert_time = time.time()

    # Draw HUD
    is_fall = fall_counter >= CONFIRM_FRAMES
    color   = (0, 0, 255) if is_fall else (0, 220, 80)
    h, w    = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 65), (15, 15, 15), -1)
    cv2.putText(frame,
                "FALL DETECTED" if is_fall else "Normal",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame,
                f"Score: {last_pred:.3f}",
                (w - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)

    # Risk bar
    bar = int((1.0 - last_pred) * (w - 40))
    cv2.rectangle(frame, (20, 68), (20+bar, 80),
                  (0,0,255) if is_fall else (0,200,80), -1)
    cv2.rectangle(frame, (20, 68), (w-20, 80), (80,80,80), 1)

    # Frame count bottom
    cv2.putText(frame,
                f"Frame: {frame_count}  |  Falls: {st.session_state.fall_count}",
                (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

    if is_fall:
        thickness = 8 if (frame_count // 5) % 2 == 0 else 3
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), thickness)

    if len(frames_buffer) < SEQUENCE_LENGTH:
        cv2.putText(frame,
                    f"Loading: {len(frames_buffer)}/{SEQUENCE_LENGTH}",
                    (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # Show in Streamlit
    st_frame.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True
    )

    # Update metrics
    fall_metric.metric("🚨 Falls", st.session_state.fall_count)
    status_metric.metric("📡 Status",
                         "FALL ⚠️" if is_fall else "Normal ✅")

    if progress_bar and total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    time.sleep(SPEED)

# Cleanup
cap.release()
if tmp_file and os.path.exists(tmp_file.name):
    os.unlink(tmp_file.name)