import streamlit as st
import tempfile
import os
import subprocess
import logging
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)


st.set_page_config(page_title="Vehicle Detection & Tracking", layout="centered")
st.title("🚗 Vehicle Detection, Tracking & Counting")

st.info("Upload a video to run detection and tracking. This version uses a pre-trained YOLOv7 model and SORT tracker.")

uploaded_file = st.file_uploader(" Upload a video file", type=["mp4", "avi", "mov"])
run_button = st.button(" Run Detection & Tracking")

if uploaded_file and run_button:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        input_path = temp_video.name

    # Define output path
    output_path = os.path.join(tempfile.gettempdir(), "output_tracked.mp4")

    # Path to YOLOv7 weights (make sure this file exists)
    weights_path = "/content/yolov7/best.pt"

    # Call the detect_track.py script
    st.info("Running detection and tracking...")
    result = subprocess.run(
        ["python3", "detect_track.py", input_path, weights_path, output_path],
        capture_output=True,
        text=True
    )

    if result.returncode == 0 and os.path.exists(output_path):
        st.success("Detection complete! Here's the output:")
        st.video(output_path)
    else:
        st.error("Detection failed. See error log below:")
        st.text(result.stderr)
