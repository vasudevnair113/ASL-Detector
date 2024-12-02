import streamlit as st
import cv2
import torch

st.set_page_config(page_title="ASL Detection", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nabla&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Commissioner:wght@100..900&family=Nabla&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Commissioner:wght@100..900&family=Nabla&family=Saira:ital,wght@0,100..900;1,100..900&display=swap');
    .stApp {
        background-color: #87CEEB;
    }
    .big-font {
        font-family: 'Saira', sans-serif;
        font-size: 48px !important;
        font-weight: bold;
        color: white;
    }
    .title {
        font-family: 'Nabla', serif;
        font-size: 70px;
        font-weight: 900;
        color: white;
    }
    .subtitle {
        font-family: 'Commissioner', sans-serif;
        font-size: 24px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/vasudevnair113/Downloads/content/yolov5/runs/train/yolov5s_results/weights/best.pt')

def detect_asl(frame):
    results = model(frame)
    return results

#st.title("ASL Detector")
st.markdown("<p class='title'>ASL Detector</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Crafted during AIM Fall 2024</p>", unsafe_allow_html=True)


col1, col2 = st.columns([2, 1])
letter_display = col2.empty()
frame_placeholder = col1.empty()
stop_button = col1.button("Stop")

cap = cv2.VideoCapture(0)

while not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detect_asl(frame_rgb)
    
    # Process detection results
    if len(results.pred[0]) > 0:
        # Get the class with highest confidence
        best_pred = results.pred[0][results.pred[0][:, 4].argmax()]
        class_id = int(best_pred[5])
        confidence = float(best_pred[4])
        letter = results.names[class_id]

        if confidence > 0.50:
            letter_display.markdown(f"<p class='big-font'>Detected Letter: {letter}</p>", unsafe_allow_html=True)
        else:
            letter_display.markdown("<p class='big-font'>No letter detected</p>", unsafe_allow_html=True)
    else:
        letter_display.markdown("<p class='big-font'>No letter detected</p>", unsafe_allow_html=True)
    
    # Display the webcam feed with a smaller size
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True, width=5)

cap.release()