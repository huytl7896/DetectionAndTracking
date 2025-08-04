#!/usr/bin/env python3
# filepath: /home/binhnguyenduc/datn/src/object_detection_pkg/object_detection_pkg/obj_tracking.py

import cv2
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
import onnxruntime as ort
import time
import torch
from types import SimpleNamespace

# Config values
video_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\Video\\People3.mp4"
conf_threshold = 0.3  # Tăng lên một chút để giảm noise
iou_threshold = 0.45
model_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\yolo11l.onnx"

# Cấu hình BYTETracker tối ưu cho tracking người
args = SimpleNamespace(
    track_thresh=0.5,      # Threshold để tạo track mới
    track_buffer=30,       # Số frame giữ track khi mất detection
    match_thresh=0.8,      # Threshold để match detection với track
    min_box_area=200,      # Diện tích tối thiểu của bounding box
    mot20=False
)

tracker = BYTETracker(args, frame_rate=30)

# COCO class names
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

person_class_id = 0  # 'person' là index 0 trong COCO classes
print(f"Person class ID: {person_class_id}")

# Tạo màu ngẫu nhiên cho tracking
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3))

def initialize_onnx_model(model_path):
    """Khởi tạo ONNX model"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Model loaded successfully")
        print(f"Input shape: {input_shape}")
        
        return session, input_name, input_shape, output_names
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        exit()

def preprocess_image(image, input_shape):
    """Tiền xử lý ảnh"""
    target_size = (input_shape[3], input_shape[2])
    
    resized = cv2.resize(image, target_size)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, target_size

def apply_nms(boxes, scores, class_ids, iou_threshold):
    """Áp dụng Non-Maximum Suppression"""
    if len(boxes) == 0:
        return [], [], []
    
    # Chuyển đổi định dạng cho OpenCV NMS
    boxes_for_nms = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        boxes_for_nms.append([x1, y1, width, height])
    
    # Áp dụng NMS
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf_threshold, iou_threshold)
    
    # Trả về các detection được giữ lại
    if len(indices) > 0:
        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]
        
        return ([boxes[i] for i in indices], 
                [scores[i] for i in indices], 
                [class_ids[i] for i in indices])
    else:
        return [], [], []

def postprocess_detections(outputs, original_shape, target_size, conf_threshold):
    """Xử lý kết quả detection với NMS"""
    try:
        predictions = outputs[0][0]  # Remove batch dimension
        
        boxes = []
        scores = []
        class_ids = []
        
        scale_x = original_shape[1] / target_size[0]
        scale_y = original_shape[0] / target_size[1]
        
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()
        
        num_detections = predictions.shape[0]
        
        for i in range(num_detections):
            try:
                detection = predictions[i]
                
                if len(detection) >= 4 + len(coco_classes):
                    x_center, y_center, width, height = detection[:4]
                    class_scores = detection[4:4+len(coco_classes)]
                    
                    class_id = int(np.argmax(class_scores))
                    max_score = float(class_scores[class_id])
                    
                    # Chỉ xử lý người (person)
                    if class_id == person_class_id and max_score > conf_threshold:
                        # Scale coordinates
                        x_center_scaled = x_center * scale_x
                        y_center_scaled = y_center * scale_y
                        width_scaled = width * scale_x
                        height_scaled = height * scale_y
                        
                        # Convert to corner coordinates
                        x1 = int(max(0, x_center_scaled - width_scaled / 2))
                        y1 = int(max(0, y_center_scaled - height_scaled / 2))
                        x2 = int(min(original_shape[1] - 1, x_center_scaled + width_scaled / 2))
                        y2 = int(min(original_shape[0] - 1, y_center_scaled + height_scaled / 2))
                        
                        # Kiểm tra kích thước hợp lệ (điều chỉnh cho người)
                        if x2 > x1 and y2 > y1 and width_scaled > 20 and height_scaled > 40:
                            boxes.append([x1, y1, x2, y2])
                            scores.append(max_score)
                            class_ids.append(class_id)
                            
            except Exception as e:
                print(f"Error processing detection {i}: {str(e)}")
                continue
        
        # Áp dụng NMS
        nms_boxes, nms_scores, nms_class_ids = apply_nms(boxes, scores, class_ids, iou_threshold)
        
        # Tạo danh sách detections cuối cùng
        detections = []
        for i in range(len(nms_boxes)):
            detections.append({
                'bbox': nms_boxes[i],
                'confidence': nms_scores[i],
                'class_id': nms_class_ids[i]
            })
        
        return detections
        
    except Exception as e:
        print(f"Error in postprocessing: {str(e)}")
        return []

def draw_safe_track(frame, track):
    """Vẽ track với error handling"""
    try:
        track_id = str(track.track_id)
        
        bbox = track.tlbr
        if not isinstance(bbox, (np.ndarray, list)) or len(bbox) != 4:
            return False
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate coordinates
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            return False
        if x1 >= frame.shape[1] or y1 >= frame.shape[0]:
            return False
            
        # Safe color selection
        color_idx = int(track_id) % len(colors)
        color = tuple(map(int, colors[color_idx]))
        
        # Draw tracking box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        track_label = f"Person #{track_id}"
        
        # Draw text with background
        label_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, track_label, (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return True
        
    except Exception as e:
        print(f"Error drawing track (ID: {getattr(track, 'track_id', 'unknown')}): {str(e)}")
        return False

# Initialize model
session, input_name, input_shape, output_names = initialize_onnx_model(model_path)

# Main processing loop
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file")
    exit()

print(f"Video opened successfully")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    try:
        # Preprocess
        input_tensor, target_size = preprocess_image(frame, input_shape)
        
        # Inference
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # Postprocess với NMS
        detections_raw = postprocess_detections(outputs, frame.shape, target_size, conf_threshold)
        
        # Chuẩn bị detections cho BYTETracker
        person_detections = []
        for det in detections_raw:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1

            # Lọc detection quá nhỏ hoặc không hợp lệ
            if width < 30 or height < 60:  # Điều chỉnh cho người (cao hơn rộng)
                continue

            confidence = float(det['confidence'])
            class_id = det['class_id']

            person_detections.append([
                float(x1), float(y1), float(x2), float(y2),
                confidence, float(class_id)
            ])

        # Chuyển đổi sang tensor
        if len(person_detections) == 0:
            person_detections_tensor = np.empty((0, 6), dtype=np.float32)
        else:
            person_detections_tensor = np.array(person_detections, dtype=np.float32)

        # Update tracker
        tracks = tracker.update(person_detections_tensor, img_info=frame.shape[:2], img_size=frame.shape[:2])
        
        # Debug information
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Frame {frame_count}: {len(person_detections)} persons detected, {len(tracks)} tracks")
        
        # Vẽ tracked objects
        drawn_tracks = 0
        tracked_objects = []
        for track in tracks:
            # BYTETracker state: 1 = Confirmed, 2 = Tracked
            if hasattr(track, "state") and track.state in [1, 2]: 
                # Thông tin tracking
                conf = getattr(track, 'det_conf', None)
                cls = getattr(track, 'det_class', None)

                tracked_objects.append({
                    'id': track.track_id,
                    'bbox': track.tlbr.tolist() if hasattr(track.tlbr, 'tolist') else track.tlbr,
                    'confidence': conf,
                    'class_id': cls
                })

                if draw_safe_track(frame, track):
                    drawn_tracks += 1
        
        # Debug tracked objects (không print mỗi frame)
        if frame_count % 30 == 0 and tracked_objects:
            print(f"Tracked objects: {len(tracked_objects)} persons")
        
        # Hiển thị thông tin trên frame
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Tracked Persons: {drawn_tracks}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error processing frame {frame_count}: {str(e)}")
    
    cv2.imshow("Person Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")