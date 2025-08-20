
import cv2
import torch
import numpy as np
import os
import time
from sort import Sort
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from IPython.display import display, clear_output
import PIL.Image

CLASSES = [
    'Indian Auto',
    'Indian Truck',
    'Bus',
    'Truck',
    'Tempo Traveller',
    'Tractor',
    'Car',
    'Two Wheeler'
]

ROI_LINE_Y = 300  

def draw_roi(frame):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, ROI_LINE_Y), (w, ROI_LINE_Y), (255, 0, 0), 2)

def crossed_line(y_old, y_new, line_y):
    return (y_old < line_y and y_new >= line_y) or (y_old > line_y and y_new <= line_y)

def main(source=0, weights='/content/yolov7/best.pt', save_path='output/tracked_output.mp4'):
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    model.eval()

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error opening video source {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0 or np.isnan(fps):
      fps = 30  # safe fallback


    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    track_hist = {}
    in_count, out_count = 0, 0
    try:
      while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (640, 640))
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

        detections, class_ids = [], []

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append([x1, y1, x2, y2, float(conf)])
                class_ids.append(int(cls))

        detections_np = np.array(detections)
        if len(detections_np):
            outputs = tracker.update(detections_np)
        else:
            outputs = np.empty((0, 5))

        draw_roi(frame)

        for det in outputs:
            if len(det) < 5:
                continue
            x1, y1, x2, y2, track_id = det[:5].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cls = -1
            for j, (dx1, dy1, dx2, dy2, conf) in enumerate(detections):
                if abs(x1 - dx1) < 10 and abs(y1 - dy1) < 10:
                    cls = class_ids[j]
                    break

            label = f"{CLASSES[cls]} ID {track_id}" if 0 <= cls < len(CLASSES) else f"ID {track_id}"

            if track_id not in track_hist:
                track_hist[track_id] = []
            track_hist[track_id].append((cx, cy))

            if len(track_hist[track_id]) >= 2:
                y_old = track_hist[track_id][-2][1]
                y_new = track_hist[track_id][-1][1]
                if crossed_line(y_old, y_new, ROI_LINE_Y):
                    if y_new > ROI_LINE_Y:
                        in_count += 1
                    else:
                        out_count += 1
                    track_hist[track_id] = []

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            for k in range(1, len(track_hist[track_id])):
                cv2.line(frame, track_hist[track_id][k - 1], track_hist[track_id][k], (0, 0, 255), 2)

        cv2.putText(frame, f"IN: {in_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"OUT: {out_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        display(PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        clear_output(wait=True)
        time.sleep(0.03)

        out.write(frame)
    finally:
      cap.release()
      out.release()

if __name__ == "__main__":
    main('/content/yolov7/test1.mp4', '/content/yolov7/best.pt', 'output/tracked_output.mp4')

