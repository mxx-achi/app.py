import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- ส่วนดึงข้อมูลพิกัด (จาก test.py) ---
class CoordTransformer:
    def __init__(self):
        self.H = None
        if os.path.exists("conveyor_calib.npz"):
            data = np.load("conveyor_calib.npz")
            self.H = data["homography"]
    
    def px_to_mm(self, x, y):
        if self.H is not None:
            pt = np.array([[[x, y]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.H)[0][0]
            return float(out[0]), float(out[1])
        return x / 3.2, y / 3.2

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
transformer = CoordTransformer()

class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5, verbose=False)
        if results and results[0].obb is not None:
            for obb in results[0].obb:
                corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                cv2.polylines(img, [corners], True, (0, 255, 100), 2)
                cx, cy = map(float, obb.xywhr[0][:2])
                x_mm, y_mm = transformer.px_to_mm(cx, cy)
                cv2.putText(img, f"{x_mm:.1f}mm", (int(cx), int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

st.title("SCARA Vision - Web Realtime")
webrtc_streamer(key="yolo", video_transformer_factory=YOLOTransformer)
