"""
Vision Detection Module - SCARA Conveyor System
กล้อง USB (OpenCV) + YOLOv8s-OBB + Homography calibration
Output: list ของ PacketInfo (x_mm, y_mm, angle) สำหรับส่งให้ SCARA#2
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
import os


# ─── Config ────────────────────────────────────────────────────────────────────

CAMERA_INDEX        = 1
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
CONFIDENCE_THRESH   = 0.5
PICK_QUEUE_TARGET   = 4

MODEL_PATH          = "best.pt"           # *** ใส่ path จริงของ best.pt ***
CAMERA_PARAMS_FILE  = "camera_params.npz" # จาก calibrate_camera.py
CONVEYOR_CALIB_FILE = "conveyor_calib.npz"# จาก calibrate_conveyor.py

# fallback ถ้าไม่มีไฟล์ calibration (ค่าประมาณ)
PX_PER_MM_FALLBACK  = 3.2


# ─── Data Class ────────────────────────────────────────────────────────────────

@dataclass
class PacketInfo:
    x_px:       float
    y_px:       float
    x_mm:       float        # พิกัดจริงบนสายพาน (mm)
    y_mm:       float
    angle:      float        # องศา จาก OBB
    confidence: float
    corners:    np.ndarray   # shape (4,2) pixel corners


# ─── Calibration Loader ────────────────────────────────────────────────────────

class CoordTransformer:
    """แปลง pixel → mm โดยใช้ homography ถ้ามี หรือ fallback px_per_mm"""

    def __init__(self):
        self.H          = None
        self.mtx        = None
        self.dist       = None
        self.new_mtx    = None
        self.px_per_mm  = PX_PER_MM_FALLBACK

        self._load_camera_params()
        self._load_conveyor_calib()

    def _load_camera_params(self):
        if not os.path.exists(CAMERA_PARAMS_FILE):
            print(f"[calib] ไม่พบ {CAMERA_PARAMS_FILE} → ข้าม undistort")
            return
        data = np.load(CAMERA_PARAMS_FILE)
        self.mtx  = data["camera_matrix"]
        self.dist = data["dist_coeffs"]
        h, w = FRAME_HEIGHT, FRAME_WIDTH
        self.new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h)
        )
        print(f"[calib] โหลด camera intrinsics จาก {CAMERA_PARAMS_FILE}")

    def _load_conveyor_calib(self):
        if not os.path.exists(CONVEYOR_CALIB_FILE):
            print(f"[calib] ไม่พบ {CONVEYOR_CALIB_FILE} → ใช้ PX_PER_MM={PX_PER_MM_FALLBACK}")
            return
        data = np.load(CONVEYOR_CALIB_FILE)
        self.H         = data["homography"]
        self.px_per_mm = float(data["px_per_mm"])
        print(f"[calib] โหลด homography จาก {CONVEYOR_CALIB_FILE}  "
              f"(px_per_mm≈{self.px_per_mm:.3f})")

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.mtx is None:
            return frame
        return cv2.undistort(frame, self.mtx, self.dist, None, self.new_mtx)

    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """แปลง pixel → mm บนระนาบสายพาน"""
        if self.H is not None:
            pt  = np.array([[[x_px, y_px]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.H)[0][0]
            return float(out[0]), float(out[1])
        # fallback: linear
        return x_px / self.px_per_mm, y_px / self.px_per_mm


# ─── Camera ────────────────────────────────────────────────────────────────────

def init_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้อง index {index} ได้")
    return cap


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray:
    cap.grab()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("อ่าน frame ไม่ได้")
    return frame


# ─── Detection ─────────────────────────────────────────────────────────────────

def detect_packets(
    model:      YOLO,
    frame:      np.ndarray,
    transformer: "CoordTransformer",
    conf_thresh: float = CONFIDENCE_THRESH,
) -> list[PacketInfo]:
    """
    YOLOv8-OBB detect → แปลง pixel เป็น mm จริงผ่าน homography
    เรียงซ้าย→ขวา (FIFO)
    """
    raw = model(frame, verbose=False)
    if not raw:
        return []
    results = raw[0]
    if results is None or results.obb is None:
        return []

    packets = []
    for obb in results.obb:
        conf = float(obb.conf)
        if conf < conf_thresh:
            continue

        cx, cy, w, h, r = map(float, obb.xywhr[0])
        angle_deg = float(np.degrees(r))
        corners   = obb.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(int)

        x_mm, y_mm = transformer.px_to_mm(cx, cy)

        packets.append(PacketInfo(
            x_px=cx, y_px=cy,
            x_mm=x_mm, y_mm=y_mm,
            angle=angle_deg,
            confidence=conf,
            corners=corners,
        ))

    packets.sort(key=lambda p: p.x_px)
    return packets


# ─── Queue Logic ───────────────────────────────────────────────────────────────

def get_pick_group(
    packets: list[PacketInfo],
    n: int = PICK_QUEUE_TARGET,
) -> tuple[list[PacketInfo], bool]:
    if len(packets) >= n:
        return packets[:n], True
    return packets, False


# ─── Visualize ─────────────────────────────────────────────────────────────────

def draw_detections(
    frame:      np.ndarray,
    packets:    list[PacketInfo],
    pick_group: list[PacketInfo],
) -> np.ndarray:
    vis      = frame.copy()
    pick_ids = {id(p) for p in pick_group}

    for p in packets:
        will_pick = id(p) in pick_ids
        color     = (0, 255, 100) if will_pick else (180, 180, 180)
        thickness = 2 if will_pick else 1

        cv2.polylines(vis, [p.corners], isClosed=True, color=color, thickness=thickness)
        cv2.circle(vis, (int(p.x_px), int(p.y_px)), 4, color, -1)

        label = (f"{p.x_mm:.1f},{p.y_mm:.1f}mm  "
                 f"{p.angle:.1f}deg  {p.confidence:.2f}")
        lx = int(p.corners[:, 0].min())
        ly = int(p.corners[:, 1].min()) - 6
        cv2.putText(vis, label, (lx, max(ly, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    ready  = len(pick_group) >= PICK_QUEUE_TARGET
    status = f"Queue: {len(packets)}  |  Pick ready: {ready}"
    cv2.putText(vis, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return vis


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ไม่พบ model: '{MODEL_PATH}'\n"
            f"แก้ MODEL_PATH บรรทัดที่ 22"
        )

    print("โหลด model...")
    model       = YOLO(MODEL_PATH)
    transformer = CoordTransformer()

    print("เปิดกล้อง...")
    cap = init_camera()

    print("เริ่ม detection  (กด q เพื่อออก)")

    while True:
        frame           = grab_frame(cap)
        frame_undist    = transformer.undistort(frame)

        packets           = detect_packets(model, frame_undist, transformer)
        pick_group, ready = get_pick_group(packets)

        if ready:
            print("── PICK READY ──")
            for i, p in enumerate(pick_group, 1):
                print(f"  [{i}] x={p.x_mm:.1f}mm  y={p.y_mm:.1f}mm"
                      f"  angle={p.angle:.1f}deg  conf={p.confidence:.2f}")
            # TODO: ส่ง pick_group ให้ SCARA#2
        else:
            print(f"รอ... {len(packets)}/{PICK_QUEUE_TARGET} ชิ้น")

        vis = draw_detections(frame_undist, packets, pick_group)
        cv2.imshow("Vision - SCARA Conveyor (OBB)", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
