#!/usr/bin/env python3
# https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import onnxruntime as ort
import time
import torch

# Config values
video_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\Video\\People3.mp4"
conf_threshold = 0.2 # Tăng lên để giảm false positives
iou_threshold = 0.4  # Threshold cho NMS
prototxt = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\Deepsort\\MobileNet\\deploy.prototxt"
model = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\Deepsort\\MobileNet\\mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

if net is not None:
    # tạo ảnh trắng test
    dummy_frame = 255 * np.ones((300, 300, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(dummy_frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    try:
        output = net.forward()
        print("[INFO] Forward pass thành công, mô hình hoạt động.")
    except Exception as e:
        print(f"[ERROR] Forward pass lỗi: {e}")

# Khởi tạo DeepSort với tham số tối ưu hơn
tracker = DeepSort(
    max_age=30,          # Giảm xuống để tracks cũ bị xóa nhanh hơn
    n_init=5,           # Tăng lên để yêu cầu nhiều detections liên tiếp hơn
    nn_budget=100,      # Giới hạn số lượng features lưu trữ
    max_cosine_distance=0.15,  # Giảm xuống để tăng tính chính xác khi matching
    max_iou_distance=0.5,     # Giảm xuống để tránh gán nhầm các tracks gần nhau
)

# COCO class names
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

# Tìm class ID của "car"
car_class_id = VOC_CLASSES.index('person')
print(f"Car class ID: {car_class_id}")

# Tạo màu ngẫu nhiên cho tracking
np.random.seed(42)  # Đảm bảo màu sắc nhất quán
colors = np.random.randint(0, 255, size=(100, 3))


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

def postprocess_detections(frame, outs, conf_threshold):
    """
    Xử lý đầu ra từ MobileNet-SSD model sau khi dự đoán.

    Parameters:
        frame (ndarray): ảnh gốc.
        outs (ndarray): đầu ra của model sau khi gọi `net.forward()`.
        conf_threshold (float): ngưỡng confidence để giữ lại các object.

    Returns:
        boxes (list): Danh sách bounding box [x, y, w, h].
        confidences (list): Danh sách confidence score.
        class_ids (list): Danh sách ID của object class.
    """
    frame_height, frame_width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for detection in outs[0, 0, :, :]:
        confidence = float(detection[2])
        if confidence > conf_threshold:
            class_id = int(detection[1])
            x1 = int(detection[3] * frame_width)
            y1 = int(detection[4] * frame_height)
            x2 = int(detection[5] * frame_width)
            y2 = int(detection[6] * frame_height)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(confidence)
            class_ids.append(class_id)

    return boxes, confidences, class_ids


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
input_shape = (300, 300)  # hoặc (320, 320) nếu đúng với model bạn dùng

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
        # Resize và tạo blob (chuẩn MobileNet SSD Caffe)
        # Trước phần xử lý ảnh
        target_size = (300, 300)

        # Resize frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, target_size), 0.007843, target_size, 127.5)

        # Set input & chạy forward
        net.setInput(blob)
        outputs = net.forward()
        
        # Postprocess với NMS để loại bỏ box trùng lặp
        boxes, confidences, class_ids = postprocess_detections(frame, outputs, conf_threshold)
        
        # Filter cars for tracking - chỉ lọc xe hơi (đã lọc sẵn trong postprocess)
        car_detections = []
        for i in range(len(boxes)):
            if class_ids[i] != car_class_id:
                continue

            x1, y1, w, h = boxes[i]
            x2 = x1 + w
            y2 = y1 + h
            confidence = confidences[i]

            if w < 40 or h < 40:
                continue

            bbox = [x1, y1, w, h]  # Format for DeepSort: [x,y,w,h]

            # Kiểm tra các bbox quá gần
            is_too_close = False
            for existing_det in car_detections:
                ex_x, ex_y, ex_w, ex_h = existing_det[0]
                dist = np.sqrt((ex_x - x1)**2 + (ex_y - y1)**2)
                if dist < 50:
                    is_too_close = True
                    break

            if not is_too_close:
                car_detections.append((bbox, confidence, class_ids[i]))
            
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