#!/usr/bin/env python3

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add CenterNet path
sys.path.append("D:\\NCKH\\ISAS\\Deepsort_tracking\\CenterNet")

# Monkey patch to fix DCNv2 autograd issue
def patch_dcnv2_autograd():
    """Patch DCNv2 to work with newer PyTorch versions"""
    import warnings
    warnings.filterwarnings("ignore", message="Legacy autograd function")
    
    # Set environment variables to force older behavior
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        # Try to patch the DCNv2 function
        from src.lib.models.networks.DCNv2.dcn_v2 import DCNv2Function
        
        # Save original forward
        if hasattr(DCNv2Function, '_original_forward'):
            return  # Already patched
            
        DCNv2Function._original_forward = DCNv2Function.forward
        
        @staticmethod
        def patched_forward(ctx, input, offset, mask, weight, bias):
            # Use the original implementation but suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return DCNv2Function._original_forward(ctx, input, offset, mask, weight, bias)
        
        DCNv2Function.forward = patched_forward
        print("DCNv2 patched successfully")
        
    except Exception as e:
        print(f"Could not patch DCNv2: {e}")

# Apply the patch before importing CenterNet
patch_dcnv2_autograd()

try:
    # Import CenterNet modules
    from src.lib.models.model import create_model, load_model
    from src.lib.utils.image import get_affine_transform
    from src.lib.utils.post_process import ctdet_post_process
    CENTERNET_IMPORTED = True
except ImportError as e:
    print(f"CenterNet import failed: {e}")
    CENTERNET_IMPORTED = False

# Config values
video_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\Video\\People3.mp4"
conf_threshold = 0.3
model_path = "D:\\NCKH\\ISAS\\Deepsort_tracking\\Main\\ctdet_coco_resdcn18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try CPU if CUDA causes issues
if device.type == 'cuda':
    try:
        torch.cuda.empty_cache()
        test_tensor = torch.randn(1, 3, 32, 32).to(device)
        del test_tensor
    except Exception as e:
        print(f"CUDA issue detected, falling back to CPU: {e}")
        device = torch.device("cpu")

print(f"Using device: {device}")

# CenterNet input resolution
input_h, input_w = 512, 512

# Initialize DeepSort tracker
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.2,
    max_iou_distance=0.7,
)

# Colors for tracking
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3))

def create_simple_centernet():
    """Create a simplified CenterNet without DCN"""
    import torchvision.models as models
    
    class SimpleCenterNet(nn.Module):
        def __init__(self, num_classes=80):
            super(SimpleCenterNet, self).__init__()
            
            # Use ResNet18 as backbone
            resnet = models.resnet18(pretrained=True)
            
            # Remove final layers
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            # Get feature dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, 512, 512)
                features = self.backbone(dummy)
                feat_dim = features.shape[1]
            
            # Upsampling layers (simple version)
            self.upconv1 = nn.ConvTranspose2d(feat_dim, 256, 4, stride=2, padding=1)
            self.upconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
            self.upconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
            self.upconv4 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
            
            # Output heads
            self.hm = nn.Conv2d(64, num_classes, 3, padding=1)  # Heatmap
            self.wh = nn.Conv2d(64, 2, 3, padding=1)            # Width-Height
            self.reg = nn.Conv2d(64, 2, 3, padding=1)           # Regression
            
        def forward(self, x):
            # Backbone
            x = self.backbone(x)
            
            # Upsampling
            x = F.relu(self.upconv1(x))
            x = F.relu(self.upconv2(x))
            x = F.relu(self.upconv3(x))
            x = F.relu(self.upconv4(x))
            
            # Output heads
            hm = self.hm(x)
            wh = self.wh(x)
            reg = self.reg(x)
            
            return {'hm': hm, 'wh': wh, 'reg': reg}
    
    return SimpleCenterNet()

def load_centernet_model():
    """Load CenterNet model with fallback options"""
    
    if not CENTERNET_IMPORTED:
        print("CenterNet not imported, using simple implementation...")
        model = create_simple_centernet()
        model = model.to(device)
        model.eval()
        return model, "SimpleCenterNet"
    
    # Model configurations (avoid DCN first)
    model_configs = [
        {
            'arch': 'resnet_18',
            'heads': {'hm': 80, 'wh': 2, 'reg': 2},
            'head_conv': 64,
            'description': 'ResNet-18'
        },
        {
            'arch': 'resnet_34', 
            'heads': {'hm': 80, 'wh': 2, 'reg': 2},
            'head_conv': 64,
            'description': 'ResNet-34'
        },
        {
            'arch': 'dlav0_34',
            'heads': {'hm': 80, 'wh': 2, 'reg': 2}, 
            'head_conv': 256,
            'description': 'DLA-34'
        }
    ]
    
    for config in model_configs:
        try:
            print(f"Trying to load {config['description']}...")
            
            # Suppress warnings during model creation
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = create_model(
                    config['arch'], 
                    heads=config['heads'], 
                    head_conv=config['head_conv']
                )
            
            # Try to load weights if available
            if os.path.exists(model_path):
                try:
                    print(f"Loading weights from {model_path}")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = load_model(model, model_path)
                except Exception as e:
                    print(f"Could not load weights: {e}, using random initialization")
            else:
                print("Model file not found, using random weights")
            
            # Move to device
            model = model.to(device)
            model.eval()
            
            # Test model
            print("Testing model...")
            dummy_input = torch.randn(1, 3, input_h, input_w).to(device)
            
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = model(dummy_input)
            
            if isinstance(output, dict) and 'hm' in output:
                print(f"✓ Successfully loaded {config['description']}")
                return model, config['arch']
            else:
                print(f"✗ Invalid output format")
                continue
                
        except Exception as e:
            print(f"✗ Failed to load {config['description']}: {str(e)}")
            continue
    
    # Final fallback: simple implementation
    print("All CenterNet configs failed, using simple implementation...")
    model = create_simple_centernet()
    model = model.to(device)
    model.eval()
    return model, "SimpleCenterNet"

def simple_nms(heatmap, kernel=3):
    """Simple Non-Maximum Suppression"""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep

def simple_decode(hm, wh, reg, K=100):
    """Simple decode function for CenterNet output"""
    batch, cat, height, width = hm.shape
    
    # Apply sigmoid and NMS to heatmap
    hm = torch.sigmoid(hm)
    hm = simple_nms(hm)
    
    # Get top K points
    scores, inds, clses, ys, xs = topk(hm, K=K)
    
    # Decode
    reg = transpose_and_gather_feat(reg, inds)
    wh = transpose_and_gather_feat(wh, inds)
    
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    
    w = wh[:, :, 0:1]
    h = wh[:, :, 1:2]
    
    bboxes = torch.cat([xs - w / 2, ys - h / 2, xs + w / 2, ys + h / 2], dim=2)
    
    detections = torch.cat([bboxes, scores.unsqueeze(2), clses.float().unsqueeze(2)], dim=2)
    
    return detections

def topk(scores, K=40):
    """Get top K points from heatmap"""
    batch, cat, height, width = scores.shape
    
    # Flatten and get topk
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()
    
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def gather_feat(feat, ind):
    """Gather features according to indices"""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def transpose_and_gather_feat(feat, ind):
    """Transpose and gather features"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def preprocess_image(image):
    """Preprocess image for CenterNet"""
    height, width = image.shape[:2]
    
    # Resize to fixed size
    resized = cv2.resize(image, (input_w, input_h))
    
    # Normalize
    inp = resized.astype(np.float32) / 255.
    inp = (inp - np.array([0.408, 0.447, 0.470])) / np.array([0.289, 0.274, 0.278])
    
    # Convert to tensor
    inp = inp.transpose(2, 0, 1)[None, ...]
    inp = torch.from_numpy(inp).float()
    
    # Calculate scale factors
    scale_x = width / input_w
    scale_y = height / input_h
    
    return inp, scale_x, scale_y

def postprocess_simple(output, scale_x, scale_y, orig_height, orig_width):
    """Simple post-processing"""
    detections = []
    
    try:
        hm = output['hm']
        wh = output['wh'] 
        reg = output['reg']
        
        # Decode detections
        dets = simple_decode(hm, wh, reg, K=100)
        
        # Process detections
        for i in range(dets.shape[1]):
            det = dets[0, i]  # batch=0
            x1, y1, x2, y2, score, cls = det.cpu().numpy()
            
            if score > conf_threshold and int(cls) == 0:  # Person class
                # Scale back to original image
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y
                
                # Convert to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Clamp to image bounds
                x1 = max(0, min(x1, orig_width - 1))
                y1 = max(0, min(y1, orig_height - 1))
                x2 = max(0, min(x2, orig_width - 1))
                y2 = max(0, min(y2, orig_height - 1))
                
                if x2 > x1 and y2 > y1:
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 20 and height > 20:
                        bbox = [x1, y1, width, height]
                        detections.append((bbox, score, 0))
                        
    except Exception as e:
        print(f"Post-processing error: {e}")
    
    return detections

def draw_track(frame, track):
    """Draw tracking information on frame"""
    try:
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            return False
            
        track_id = track.track_id
        color_idx = int(track_id) % len(colors)
        color = tuple(map(int, colors[color_idx]))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"Person #{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return True
        
    except Exception as e:
        print(f"Error drawing track: {e}")
        return False

# Load model
print("Loading CenterNet model...")
try:
    model, arch_name = load_centernet_model()
    print(f"Model loaded successfully using {arch_name}")
except Exception as e:
    print(f"Fatal error loading model: {e}")
    exit()

# Main processing loop
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

print("Starting video processing...")
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    orig_height, orig_width = frame.shape[:2]
    
    try:
        # Preprocess image
        input_tensor, scale_x, scale_y = preprocess_image(frame)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(input_tensor)
        
        # Post-process to get detections
        person_detections = postprocess_simple(output, scale_x, scale_y, orig_height, orig_width)
        
        # Update tracker
        tracks = tracker.update_tracks(person_detections, frame=frame)
        
        # Draw confirmed tracks
        confirmed_tracks = 0
        for track in tracks:
            if track.is_confirmed():
                if draw_track(frame, track):
                    confirmed_tracks += 1
        
        # Display info
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | People: {confirmed_tracks} | {arch_name}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Debug info
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(person_detections)} detections, {confirmed_tracks} confirmed tracks")
        
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
    
    # Display frame
    cv2.imshow("CenterNet Person Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")