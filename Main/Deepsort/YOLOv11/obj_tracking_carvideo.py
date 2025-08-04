#!/usr/bin/env python3

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import onnxruntime as ort
import time
import torch

# Config values
video_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Deepsort_tracking\\People3.mp4"
conf_threshold = 0.2 # Tăng lên để giảm false positives
iou_threshold = 0.4  # Threshold cho NMS
model_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Deepsort_tracking\\yolo11l.onnx"

# Khởi tạo DeepSort với tham số tối ưu hơn
tracker = DeepSort(
    max_age=30,          # Giảm xuống để tracks cũ bị xóa nhanh hơn
    n_init=5,           # Tăng lên để yêu cầu nhiều detections liên tiếp hơn
    nn_budget=100,      # Giới hạn số lượng features lưu trữ
    max_cosine_distance=0.15,  # Giảm xuống để tăng tính chính xác khi matching
    max_iou_distance=0.5,     # Giảm xuống để tránh gán nhầm các tracks gần nhau
)

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

# Tìm class ID của "car"
car_class_id = coco_classes.index('person')
print(f"Car class ID: {car_class_id}")

# Tạo màu ngẫu nhiên cho tracking
np.random.seed(42)  # Đảm bảo màu sắc nhất quán
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

def apply_nms(boxes, scores, iou_threshold):
    """Áp dụng Non-Maximum Suppression để loại bỏ box trùng lặp"""
    # Chuyển đổi định dạng để phù hợp với hàm NMSBoxes của OpenCV
    boxes_for_nms = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        boxes_for_nms.append([x1, y1, width, height])
    
    # Áp dụng NMS
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf_threshold, iou_threshold)
    
    # Trả về các box được giữ lại
    if len(indices) > 0:
        if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
            # OpenCV < 4.5.4
            return [boxes[i[0]] for i in indices]
        else:
            # OpenCV >= 4.5.4
            return [boxes[i] for i in indices]
    else:
        return []

def postprocess_detections(outputs, original_shape, target_size, conf_threshold):
    """Xử lý kết quả detection với error handling và NMS"""
    try:
        predictions = outputs[0][0]  # Remove batch dimension
        
        detections = []
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
                    
                    if 0 <= class_id < len(coco_classes) and max_score > conf_threshold:
                        # Chỉ xử lý xe hơi
                        if class_id == car_class_id:
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
                            
                            if x2 > x1 and y2 > y1 and width_scaled > 30 and height_scaled > 30:
                                boxes.append([x1, y1, x2, y2])
                                scores.append(max_score)
                                class_ids.append(class_id)
                            
            except Exception as e:
                print(f"Error processing detection {i}: {str(e)}")
                continue
        
        # Áp dụng NMS để loại bỏ box trùng lặp
        if len(boxes) > 0:
            nms_boxes = apply_nms(boxes, scores, iou_threshold)
            
            # Tạo detections từ các box sau NMS
            for i, box in enumerate(boxes):
                if box in nms_boxes:
                    detections.append({
                        'bbox': box,
                        'confidence': scores[i],
                        'class_id': class_ids[i]
                    })
        
        return detections
        
    except Exception as e:
        print(f"Error in postprocessing: {str(e)}")
        return []

def draw_safe_track(frame, track):
    """Vẽ track với extra debugging và error handling"""
    try:
        # Get track ID and convert to string explicitly
        track_id = str(track.track_id)
        
        # Get bounding box with extra validation
        try:
            bbox = track.to_tlbr()
            
            # Validate bbox format
            if not isinstance(bbox, np.ndarray) and not isinstance(bbox, list):
                print(f"Invalid bbox type: {type(bbox)}")
                return False
                
            if len(bbox) != 4:
                print(f"Invalid bbox length: {len(bbox)}")
                return False
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Validate bbox coordinates
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
                print(f"Invalid bbox coordinates: {x1},{y1},{x2},{y2}")
                return False
                
        except Exception as bbox_error:
            print(f"Error getting bbox: {str(bbox_error)}")
            return False
            
        # Safe color selection
        color_idx = int(track_id) % len(colors)
        color = tuple(map(int, colors[color_idx]))
        
        # Draw tracking box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw track ID
        track_label = f"Person #{track_id}"
        
        # Draw text with background
        label_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, track_label, (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return True  # Successfully drawn
        
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
print(torch.__version__)
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
        
        # Postprocess với NMS để loại bỏ box trùng lặp
        detections_raw = postprocess_detections(outputs, frame.shape, target_size, conf_threshold)
        
        # Filter cars for tracking - chỉ lọc xe hơi (đã lọc sẵn trong postprocess)
        car_detections = []
        for det in detections_raw:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
                
            # Skip small detections
            if width < 40 or height < 40:
                continue
                
            bbox = [x1, y1, width, height]  # Format for DeepSort: [x,y,w,h]
            confidence = det['confidence']

            is_too_close = False
            for existing_det in car_detections:
                existing_bbox = existing_det[0]  # [x, y, w, h]
                # Calculate IoU between current bbox and existing bbox
                ex_x, ex_y, ex_w, ex_h = existing_bbox
                current_x, current_y, current_w, current_h = bbox
                
                # Calculate centers
                current_center_x = current_x + current_w/2
                current_center_y = current_y + current_h/2
                ex_center_x = ex_x + ex_w/2
                ex_center_y = ex_y + ex_h/2
                
                # Calculate distance between centers
                distance = np.sqrt((current_center_x - ex_center_x)**2 + (current_center_y - ex_center_y)**2)
                
                # If centers are too close, skip this detection
                if distance < 50:  # Adjust this threshold as needed
                    is_too_close = True
                    break
            
            if not is_too_close:
                # DeepSort expects (bbox, confidence, feature) or (bbox, confidence, class_id)
                car_detections.append((bbox, confidence, int(car_class_id)))
        
        # Update tracks với chỉ xe hơi
        tracks = tracker.update_tracks(car_detections, frame=frame)
        #print(f"Tracks: {tracks}")
        
        # Debug information
        if frame_count % 20 == 0:  # Print every 20 frames
            print(f"Frame {frame_count}: {len(car_detections)} people detected, {len(tracks)} tracks, {len([t for t in tracks if t.is_confirmed()])} confirmed tracks")
        
        # Chỉ vẽ các đối tượng đã xác nhận (confirmed)
        #lưu thông tin các traking object hiện tại bao gồm cả id track

        drawn_tracks = 0
        tracked_objects = []
        for track in tracks:
            if track.is_confirmed():
                # Độ tin cậy và class_id – bảo đảm truy cập an toàn
                conf = track.det_conf if hasattr(track, 'det_conf') else None
                cls  = track.det_class if hasattr(track, 'det_class') else None

                tracked_objects.append({
                    'id'        : track.track_id,
                    'bbox'      : track.to_tlbr(),
                    'confidence': conf,
                    'class_id'  : cls
                })

                if draw_safe_track(frame, track):
                    drawn_tracks += 1
        print(f"Tracked objects at frame: {tracked_objects}")
        
        # Hiển thị thông tin
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Tracked People: {drawn_tracks}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error processing frame {frame_count}: {str(e)}")
    
    cv2.imshow("People Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")