import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import math

# إعداد Streamlit
st.set_page_config(page_title="Swimmer Analysis", layout="wide")
st.title("🏊‍♂️ Swimmer Analysis")
st.markdown("---")


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

    if total_time > 0:
        speed = distance_in_meters / total_time
        return speed
    else:
        return 0.0


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
                'display_id': None,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf
            })

        if len(detections) >= expected_min:
            chosen_model = model
            break

    for i, d in enumerate(detections, start=1):
        d['display_id'] = i

    return detections, chosen_model


def track_people(video_path, model, selected_bbox, pool_length, match_iou_threshold=0.4, search_frames=10):
    tracked_results = []
    target_positions = []
    target_track_id = None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = model.track(source=video_path, tracker="bytetrack.yaml", stream=True, verbose=False)

    frame_count = 0
    for result in results:
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Analysis... {frame_count}/{total_frames}")

        frame_people = []
        boxes = result.boxes
        if boxes is not None:
            for b_idx, box in enumerate(boxes):
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
    progress_bar.empty()
    status_text.empty()

    return tracked_results, target_positions, fps, target_track_id


def analyze_swimmer_speed(video_path, pool_length, selected_bbox, model):
    tracked_people, target_positions, fps, found_track_id = track_people(video_path, model, selected_bbox, pool_length)
    return tracked_people, target_positions, fps, found_track_id


# واجهة Streamlit
st.markdown("Setting")

pool_length = st.number_input(
    "lenth of pool (m):",
    min_value=10.0,
    max_value=100.0,
    value=25.0,
    step=0.5,
    help="Enter the lenth of pool (m)"
)

expected_min = st.number_input(
    "(expected count of players)",
    min_value=1,
    max_value=50,
    value=1,
    step=1
)

video_file = st.file_uploader(
    "📹 Upload Video:",
    type=["mp4", "mov", "avi", "mkv"],
    help="MP4, MOV, AVI, MKV"
)

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()

    try:
        cap = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            st.error("Cant see first frame")
        else:
            detections, model = detect_with_fallback(first_frame, expected_min=expected_min)
            if not detections:
                st.error("No Detection")
            else:
                display_frame = first_frame.copy()
                for det in detections:
                    bbox = det['bbox']
                    disp_id = det['display_id']
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID {disp_id}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Select Player ID")

                options = [f"ID {d['display_id']} - bbox {d['bbox']}" for d in detections]
                choice = st.selectbox("Select Player", options)

                if st.button("Start Analysis", type="primary"):
                    sel_idx = options.index(choice)
                    selected_bbox = detections[sel_idx]['bbox']

                    with st.spinner("Tracking by bbox..."):
                        tracked_people, target_positions, fps, found_track_id = analyze_swimmer_speed(
                            tfile.name, pool_length, selected_bbox, model
                        )

                    if target_positions and len(target_positions) > 0:
                        speed = calculate_speed_from_positions(target_positions, pool_length, fps)

                        st.markdown("### Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🏃‍♂️ Speed", f"{speed:.2f}", "m/s")
                        with col2:
                            st.metric("Analysis Time",
                                      f"{target_positions[-1]['time'] - target_positions[0]['time']:.1f}", "Second")
                        with col3:
                            st.metric("Frames Tracked", f"{len(target_positions)}")

                        st.success(
                            f"Matched tracker ID: {found_track_id}" if found_track_id is not None else "No tracker ID matched (but positions collected)."
                        )
                    else:
                        st.error("Error")

    except Exception as e:
        st.error(f"Error: {str(e)}")

    try:
        os.unlink(tfile.name)
    except Exception:
        pass

else:
    st.info("Please Upload Video to analysis")

st.markdown("---")
