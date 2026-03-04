import os
import json
import urllib.request
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")   # local machine — no CUDA


#  CNN model 

def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


class ASLClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.40):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features, momentum=0.01, eps=1e-3),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class ASLPredictor:
    def __init__(self, checkpoint_path, map_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        cfg  = ckpt["cfg"]

        self.model = ASLClassifier(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.transform = build_transforms(cfg["img_size"])

        with open(map_path) as f:
            self.class_map = {int(k): v for k, v in json.load(f).items()}

    @torch.inference_mode()
    def predict(self, pil_image):
        # accepts a PIL image directly — no need to save/reload from disk
        tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=-1)[0]
        idx    = probs.argmax().item()
        return self.class_map[idx], probs[idx].item()


# MediaPipe hand detection 

# download the hand landmarker model if not already present
MODEL_PATH = r"C:/Users/rishi/VS CODINGS/SG-1_DC_Project/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options                     = base_options,
    running_mode                     = vision.RunningMode.IMAGE,
    num_hands                        = 1,
    min_hand_detection_confidence    = 0.5,
    min_hand_presence_confidence     = 0.5,
    min_tracking_confidence          = 0.5,
)
detector = vision.HandLandmarker.create_from_options(options)


def detect_and_crop_hand(image_path, padding=20):
    img      = cv2.imread(image_path)
    h, w     = img.shape[:2]
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        print(f"No hand detected — using full image")
        return Image.fromarray(rgb)

    lm       = result.hand_landmarks[0]
    x_coords = [l.x * w for l in lm]
    y_coords = [l.y * h for l in lm]

    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_max = min(h, int(max(y_coords)) + padding)

    crop = img[y_min:y_max, x_min:x_max]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# ── run ───────────────────────────────────────────────────────────────────────

CHECKPOINT = r"C:/Users/rishi/VS CODINGS/SG-1_DC_Project/cnn_model.pth"
CLASS_MAP  = r"C:/Users/rishi/VS CODINGS/SG-1_DC_Project/class_map.json"
IMAGE_PATH = r"C:/Users/rishi/VS CODINGS/SG-1_DC_Project/F_sign.png"

predictor = ASLPredictor(CHECKPOINT, CLASS_MAP)
cropped   = detect_and_crop_hand(IMAGE_PATH)

# show original vs crop
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(Image.open(IMAGE_PATH))
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(cropped)
axes[1].set_title("Cropped hand")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# predict directly from the cropped PIL image — no temp file needed
label, conf = predictor.predict(cropped)
print(f"Predicted: {label}  ({conf*100:.1f}%)")

# cleanup
detector.close()