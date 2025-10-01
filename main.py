from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import tempfile
import os
import math
from ultralytics import YOLO

app = FastAPI(title="Swimmer Speed API")

# ------------------------
# Utilities
# ------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = (boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def calculate_speed_from_positions(positions, pool_length, fps):
    if len(positions) < 2:
        return 0.0
    total_distance = 0.0
    for i in range(1, len(positions)):
        dx = positions[i]['x'] - positions[i - 1]['x']
        dy = positions[i]['y'] - positions[i - 1]['y']
        distance = math.sqrt(dx * dx + dy * dy)
        total_distance += distance
    pixels_per_meter = 1920 / pool_length
    distance_in_meters = total_distance / pixels_per_meter
    total_time = (positions[-1]['time'] - positions[0]['time'])
    return distance_in_meters / total_time if total_time > 0 else 0.0


def detect_with_fallback(frame, expected_min=1):
    models = ["yolov8n-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt"]
    detections = []
    chosen_model = None

    for model_path in models:
        try:
            model = YOLO(model_path)
        except Exception:
            continue

        results = model.predict(frame, classes=[0], verbose=False)
        if not results:
            continue

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue

        detections = []
        for idx, box in enumerate(boxes):
            conf = float(box.conf[0].cpu().numpy())
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'display_id': idx + 1,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf
            })

        if len(detections) >= expected_min:
            chosen_model = model
            break

    return detections, chosen_model


def track_people(video_path, model, selected_bbox, pool_length, match_iou_threshold=0.4, search_frames=10):
    tracked_results = []
    target_positions = []
    target_track_id = None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    results = model.track(source=video_path, tracker="bytetrack.yaml", stream=True, verbose=False)

    frame_count = 0
    for result in results:
        frame_count += 1
        frame_people = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) != 0:
                    continue
                conf = float(box.conf[0].cpu().numpy())
                if conf < 0.2:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

                bbox_curr = [int(x1), int(y1), int(x2), int(y2)]
                frame_people.append({
                    'id': track_id,
                    'bbox': bbox_curr,
                    'confidence': conf
                })

                if target_track_id is None and frame_count <= search_frames:
                    score = iou(selected_bbox, bbox_curr)
                    if score >= match_iou_threshold:
                        target_track_id = track_id
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        target_positions.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'x': center_x,
                            'y': center_y
                        })

                if target_track_id is not None and track_id == target_track_id:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    target_positions.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'x': center_x,
                        'y': center_y
                    })

        tracked_results.append(frame_people)

    cap.release()
    return tracked_results, target_positions, fps, target_track_id


# ------------------------
# API Endpoints
# ------------------------
@app.post("/detect/")
async def detect_players(video: UploadFile, expected_min: int = Form(1)):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(await video.read())
    tfile.close()

    try:
        cap = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            return JSONResponse({"error": "Cannot read first frame"}, status_code=400)

        detections, model = detect_with_fallback(first_frame, expected_min=expected_min)
        if not detections:
            return JSONResponse({"error": "No detections found"}, status_code=404)

        return {"detections": detections}

    finally:
        os.unlink(tfile.name)


@app.post("/analyze/")
async def analyze_swimmer(video: UploadFile, pool_length: float = Form(25.0), bbox: str = Form(...)):
    """
    bbox = "x1,y1,x2,y2"
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(await video.read())
    tfile.close()

    try:
        cap = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            return JSONResponse({"error": "Cannot read first frame"}, status_code=400)

        detections, model = detect_with_fallback(first_frame)
        if not detections or model is None:
            return JSONResponse({"error": "No detections found"}, status_code=404)

        selected_bbox = list(map(int, bbox.split(",")))
        tracked_people, target_positions, fps, found_track_id = track_people(
            tfile.name, model, selected_bbox, pool_length
        )

        if target_positions:
            speed = calculate_speed_from_positions(target_positions, pool_length, fps)
            return {
                "status": "success",
                "pool_length": pool_length,
                "fps": fps,
                "frames_tracked": len(target_positions),
                "speed_m_s": round(speed, 2),
                "matched_id": found_track_id
            }
        else:
            return JSONResponse({"error": "No positions tracked"}, status_code=404)

    finally:
        os.unlink(tfile.name)
