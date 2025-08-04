#!/usr/bin/env python3

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from backbone import EfficientDetBackbone
import time
import torch
import sys
sys.path.append("D:\\NCKH\\ISAS\\Deepsort_tracking\\Yet-Another-EfficientDet-Pytorch")
from utils.utils import postprocess
from efficientdet.utils import BBoxTransform, ClipBoxes

# Config values
video_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\Video\\People3.mp4"
conf_threshold = 0.2 # Tăng lên để giảm false positives
iou_threshold = 0.4  # Threshold cho NMS

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

model_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\efficientdet-d0.pth"
compound_coef = 0  # hoặc 1, 2... tùy mô hình bạn training là D0, D1...
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90,
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(model_path, map_location='cuda'))  # hoặc 'cuda' nếu dùng GPU
model.eval()

# Khởi tạo DeepSort với tham số tối ưu hơn
tracker = DeepSort(
    max_age=30,          # Giảm xuống để tracks cũ bị xóa nhanh hơn
    n_init=5,           # Tăng lên để yêu cầu nhiều detections liên tiếp hơn
    nn_budget=100,      # Giới hạn số lượng features lưu trữ
    max_cosine_distance=0.15,  # Giảm xuống để tăng tính chính xác khi matching
    max_iou_distance=0.5,     # Giảm xuống để tránh gán nhầm các tracks gần nhau
)

# COCO class names
coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# Tìm class ID của "person"
person_class_id = coco_classes.index('person')
print(f"Person class ID: {person_class_id}")

# Tạo màu ngẫu nhiên cho tracking
np.random.seed(42)  # Đảm bảo màu sắc nhất quán
colors = np.random.randint(0, 255, size=(100, 3))

def initialize_pytorch_model(model_path, compound_coef=0, num_classes=90, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Khởi tạo PyTorch model từ file .pth"""
    # Khởi tạo mô hình
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes)
    
    # Tải trọng số từ file .pth
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Đưa về eval mode
    model.to(device)
    model.eval()

    # Tạo dummy input để kiểm tra input shape
    dummy_input = torch.randn(1, 3, 512, 512).to(device)  # Sửa 512x512 theo input model
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        print("Model loaded successfully")
        print(f"Dummy input shape: {dummy_input.shape}")
        return model, dummy_input.shape
    except Exception as e:
        print(f"Error during forward pass: {e}")
        exit()

def preprocess_image(image, input_shape):
    """Tiền xử lý ảnh"""
    target_size = (input_shape[3], input_shape[2])
    
    resized = cv2.resize(image, target_size)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, target_size

def parse_postprocess_output(outputs, original_shape, target_size):
    """Parse the output from postprocess function correctly with proper coordinate scaling"""
    detections_raw = []
    
    try:
        # Check if outputs is a list and has at least one element
        if not outputs or len(outputs) == 0:
            return detections_raw
            
        output_batch = outputs[0]  # Get first batch
        
        # Calculate scaling factors
        scale_x = original_shape[1] / target_size[0]  # width scaling
        scale_y = original_shape[0] / target_size[1]  # height scaling
        
        print(f"Original shape: {original_shape}, Target size: {target_size}")
        print(f"Scale factors - X: {scale_x:.3f}, Y: {scale_y:.3f}")
        
        # Handle different output formats
        if isinstance(output_batch, dict):
            # Dictionary format: {'rois': ..., 'class_ids': ..., 'scores': ...}
            rois = output_batch.get('rois', [])
            class_ids = output_batch.get('class_ids', [])
            scores = output_batch.get('scores', [])
            
            # Convert tensors to numpy if needed
            if isinstance(rois, torch.Tensor):
                rois = rois.cpu().numpy()
            if isinstance(class_ids, torch.Tensor):
                class_ids = class_ids.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
                
            print(f"Found {len(rois)} ROIs, {len(class_ids)} class_ids, {len(scores)} scores")
                
            # Process each detection
            if len(rois) > 0 and len(class_ids) > 0 and len(scores) > 0:
                for i in range(min(len(rois), len(class_ids), len(scores))):
                    roi = rois[i]
                    class_id = int(class_ids[i])
                    score = float(scores[i])
                    
                    if class_id == person_class_id and score > conf_threshold:
                        if len(roi) >= 4:
                            # Try different coordinate formats
                            # Format 1: [x1, y1, x2, y2] - already scaled
                            x1, y1, x2, y2 = roi[:4]
                            
                            # If coordinates are normalized (0-1), scale them
                            if max(x1, y1, x2, y2) <= 1.0:
                                x1 *= original_shape[1]
                                y1 *= original_shape[0] 
                                x2 *= original_shape[1]
                                y2 *= original_shape[0]
                            # If coordinates are in model input space, scale to original
                            elif max(x1, y1, x2, y2) <= max(target_size):
                                x1 *= scale_x
                                y1 *= scale_y
                                x2 *= scale_x
                                y2 *= scale_y
                            
                            # Convert to integers and ensure proper order
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            
                            # Ensure proper coordinate order
                            if x1 > x2:
                                x1, x2 = x2, x1
                            if y1 > y2:
                                y1, y2 = y2, y1
                            
                            # Clamp to image boundaries
                            x1 = max(0, min(x1, original_shape[1] - 1))
                            y1 = max(0, min(y1, original_shape[0] - 1))
                            x2 = max(0, min(x2, original_shape[1] - 1))
                            y2 = max(0, min(y2, original_shape[0] - 1))
                            
                            # Skip invalid boxes
                            if x2 > x1 and y2 > y1:
                                bbox = [x1, y1, x2, y2]
                                detections_raw.append({
                                    'bbox': bbox,
                                    'confidence': score,
                                    'class_id': class_id
                                })
                                print(f"Detection {i}: bbox={bbox}, conf={score:.3f}")
                                
        elif isinstance(output_batch, (list, tuple)) or isinstance(output_batch, torch.Tensor):
            # List/tensor format
            if isinstance(output_batch, torch.Tensor):
                output_batch = output_batch.cpu().numpy()
                
            for i, det in enumerate(output_batch):
                if isinstance(det, torch.Tensor):
                    det = det.cpu().numpy()
                    
                if len(det) >= 6:  # [x1, y1, x2, y2, score, class_id]
                    x1, y1, x2, y2, score, class_id = det[:6]
                    class_id = int(class_id)
                    score = float(score)
                    
                    if class_id == person_class_id and score > conf_threshold:
                        # Scale coordinates if needed
                        if max(x1, y1, x2, y2) <= 1.0:
                            x1 *= original_shape[1]
                            y1 *= original_shape[0]
                            x2 *= original_shape[1] 
                            y2 *= original_shape[0]
                        elif max(x1, y1, x2, y2) <= max(target_size):
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y
                        
                        # Convert and validate
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Clamp to boundaries
                        x1 = max(0, min(x1, original_shape[1] - 1))
                        y1 = max(0, min(y1, original_shape[0] - 1))
                        x2 = max(0, min(x2, original_shape[1] - 1))
                        y2 = max(0, min(y2, original_shape[0] - 1))
                        
                        if x2 > x1 and y2 > y1:
                            bbox = [x1, y1, x2, y2]
                            detections_raw.append({
                                'bbox': bbox,
                                'confidence': score,
                                'class_id': class_id
                            })
                            print(f"Detection {i}: bbox={bbox}, conf={score:.3f}")
        else:
            print(f"Unexpected output format: {type(output_batch)}")
            
    except Exception as e:
        print(f"Error parsing postprocess output: {str(e)}")
        print(f"Output type: {type(outputs)}")
        if outputs:
            print(f"First element type: {type(outputs[0])}")
        import traceback
        traceback.print_exc()
            
    return detections_raw

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
model, input_shape = initialize_pytorch_model(model_path, compound_coef=0, num_classes=90)

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
        input_tensor = torch.from_numpy(input_tensor).float().to(device)
        
        # Inference
        model.eval()

        # Khởi tạo các hàm xử lý bbox
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        # Trong vòng lặp xử lý từng frame:
        with torch.no_grad():
            features, regression, classification, anchors = model(input_tensor)

            # Decode output
            outputs = postprocess(
                input_tensor,
                anchors,
                regression,
                classification,
                regressBoxes,
                clipBoxes,
                threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
        
        # Parse the postprocess output correctly with coordinate scaling
        detections_raw = parse_postprocess_output(outputs, frame.shape, target_size)
                
        # Filter people for tracking
        person_detections = []
        for det in detections_raw:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
                
            # Skip small detections
            if width < 40 or height < 40:
                continue
                
            bbox = [x1, y1, width, height]  # Format for DeepSort: [x,y,w,h]
            confidence = det['confidence']
            class_id = det['class_id']

            is_too_close = False
            for existing_det in person_detections:
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
                person_detections.append((bbox, confidence, class_id))
        
        # Update tracks với chỉ người
        tracks = tracker.update_tracks(person_detections, frame=frame)
        
        # Debug information
        if frame_count % 20 == 0:  # Print every 20 frames
            print(f"Frame {frame_count}: {len(person_detections)} people detected, {len(tracks)} tracks, {len([t for t in tracks if t.is_confirmed()])} confirmed tracks")
        
        # Debug information - show detection boxes before tracking
        if frame_count % 20 == 0 and len(detections_raw) > 0:
            print(f"Frame {frame_count}: {len(detections_raw)} raw detections found")
            for i, det in enumerate(detections_raw[:3]):  # Show first 3 detections
                bbox = det['bbox']
                print(f"  Detection {i}: bbox={bbox}, conf={det['confidence']:.3f}")
        
        # Draw raw detections for debugging (optional - comment out if not needed)
        for det in detections_raw:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow boxes for raw detections
        
        # Chỉ vẽ các đối tượng đã xác nhận (confirmed)
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
        
        if frame_count % 20 == 0:  # Print tracked objects every 20 frames
            print(f"Tracked objects at frame {frame_count}: {len(tracked_objects)} objects")
        
        # Hiển thị thông tin
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Tracked People: {drawn_tracks}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error processing frame {frame_count}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    cv2.imshow("People Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")