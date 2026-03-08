"""Mediapipe crops images to extract hand => cnn extracts feature vector => lstm predicts sign using feature vectors as inputs"""

import os
import time
import json
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

from PIL import Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

#  CONFIG


CFG = {
    # paths 
    "data_dir"       : "/kaggle/input/datasets/rishisujitham25b128/asl-dynamic/ASL_data",
    "cnn_checkpoint" : "/kaggle/input/models/rishisujitham25b128/asl-cnn-model/pytorch/default/1/cnn_model.pth",   # static-sign CNN
    "output_dir"     : "/kaggle/working/lstm_output",

    # hand-crop 
    "model_path"     : "/kaggle/input/datasets/rishisujitham25b128/hand-landmarker/hand_landmarker.task",
    "crop_padding"   : 20,                                             # px padding around bbox
    "min_hand_conf"  : 0.4,                                            # lower = more lenient

    #data 
    "num_frames"     : 16,
    "img_size"       : 260,    # EfficientNet-B2 native size

    # model 
    "cnn_feature_dim": 1408,   # EfficientNet-B2 penultimate-layer width
    "proj_dim"       : 512,
    "hidden_size"    : 256,
    "num_layers"     : 2,
    "dropout"        : 0.40,
    "num_classes"    : 26,     # auto-updated from dataset

    # training 
    "epochs"         : 40,
    "batch_size"     : 30,
    "num_workers"    : 4,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-4,
    "label_smoothing": 0.10,
    "patience"       : 8,
    "seed"           : 42,
    "use_amp"        : True,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
print(f"[INFO] Device    : {DEVICE}")
print(f"[INFO] GPU count : {torch.cuda.device_count()}")


# 
#  MEDIAPIPE HAND CROPPER  (shared across workers via lazy init)
# 

assert os.path.exists(CFG["model_path"]), (
    f"hand_landmarker.task not found at {CFG['model_path']}. "
    "Upload it as a Kaggle dataset and attach it to this notebook."
)


class HandCropper:
    """
    Lazily-initialised MediaPipe hand detector.

    One instance is created per DataLoader worker (via worker_init_fn) so we
    never share the C++ detector across processes.
    """

    _instance: "HandCropper | None" = None   # per-process singleton

    def __init__(self, model_path: str, padding: int, min_conf: float):
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options                  = base_opts,
            running_mode                  = vision.RunningMode.IMAGE,
            num_hands                     = 1,
            min_hand_detection_confidence = min_conf,
            min_hand_presence_confidence  = min_conf,
            min_tracking_confidence       = min_conf,
        )
        self.detector = vision.HandLandmarker.create_from_options(opts)
        self.padding  = padding

    @classmethod
    def get(cls, cfg: dict) -> "HandCropper":
        """Return the per-process singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls(
                model_path = cfg["model_path"],
                padding    = cfg["crop_padding"],
                min_conf   = cfg["min_hand_conf"],
            )
        return cls._instance

    def crop(self, pil_img: Image.Image) -> Image.Image:
        """
        Detect the hand in *pil_img* and return a square-padded crop.
        Falls back to the full image when no hand is found.
        """
        rgb      = np.array(pil_img)                       # (H, W, 3) 
        h, w     = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return pil_img   # no hand found – use full frame

        lm       = result.hand_landmarks[0]
        xs       = [l.x * w for l in lm]
        ys       = [l.y * h for l in lm]
        x0 = max(0, int(min(xs)) - self.padding)
        y0 = max(0, int(min(ys)) - self.padding)
        x1 = min(w, int(max(xs)) + self.padding)
        y1 = min(h, int(max(ys)) + self.padding)

        crop = rgb[y0:y1, x0:x1]
        return Image.fromarray(crop)


# 
#  DATASET
# 

class DynamicASLDataset(Dataset):
    """
   

    For every clip, `num_frames` frames are sampled uniformly.
    Each frame is hand-cropped with MediaPipe before the image transform.
    """

    def __init__(self, data_dir: str, num_frames: int, transform, cfg: dict):
        self.num_frames = num_frames
        self.transform  = transform
        self.cfg        = cfg
        self.samples: list[tuple[list[str], int]] = []
        self.classes    = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for label in self.classes:
            label_dir = os.path.join(data_dir, label)
            for clip in sorted(os.listdir(label_dir)):
                clip_dir = os.path.join(label_dir, clip)
                if not os.path.isdir(clip_dir):
                    continue
                frames = sorted([
                    os.path.join(clip_dir, f)
                    for f in os.listdir(clip_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                if len(frames) >= 2:
                    self.samples.append((frames, self.class_to_idx[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frames(self, paths: list[str]) -> list[str]:
        indices = np.linspace(0, len(paths) - 1, self.num_frames, dtype=int)
        return [paths[i] for i in indices]

    def __getitem__(self, idx: int):
        frame_paths, label = self.samples[idx]
        selected           = self._sample_frames(frame_paths)

        cropper = HandCropper.get(self.cfg)   # lazily created per worker
        frames  = []
        for path in selected:
            img     = Image.open(path).convert("RGB")
            cropped = cropper.crop(img)        # ← hand-region crop
            frames.append(self.transform(cropped))

        clip_tensor = torch.stack(frames, dim=0)   # (T, C, H, W)
        return clip_tensor, label


def build_transforms(img_size: int, split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_dataloaders(cfg: dict):
    data_dir  = cfg["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    # gracefully fall back to a flat dataset with no train/test split
    if not os.path.isdir(train_dir):
        train_dir = test_dir = data_dir

    train_ds = DynamicASLDataset(train_dir, cfg["num_frames"], build_transforms(cfg["img_size"], "train"), cfg)
    test_ds  = DynamicASLDataset(test_dir,  cfg["num_frames"], build_transforms(cfg["img_size"], "test"),  cfg)

    cfg["num_classes"] = len(train_ds.classes)
    print(f"[INFO] Classes ({cfg['num_classes']}) : {train_ds.classes}")
    print(f"[INFO] Train clips : {len(train_ds):,}")
    print(f"[INFO] Test  clips : {len(test_ds):,}")

    # MediaPipe creates its own file handles per process, so we use
    # spawn-safe persistent_workers only when num_workers > 0.
    nw = cfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    return train_loader, test_loader, train_ds.classes


#  CNN    same architecture as the static-sign training script


class ASLClassifier(nn.Module):
    """
   

    Loads the weights from the cnn and extracts backbone cleanly
    """
    def __init__(self, num_classes: int, dropout: float = 0.40):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=None)
        in_features   = self.backbone.classifier[1].in_features   # 1408
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_cnn_backbone(checkpoint_path: str) -> nn.Module:
    """
    Load the full ASLClassifier from checkpoint_path, then detach and
    return only the backbone (EfficientNet-B2 up to the penultimate layer).
    The backbone classifier is already nn.Identity(), so its output is the
    raw 1408-dim feature vector we want to feed into the LSTM.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]

    full_model = ASLClassifier(
        num_classes = cfg["num_classes"],
        dropout     = cfg["dropout"],
    )
    full_model.load_state_dict(ckpt["model_state"])   # strict=True 
    print(f"[INFO] CNN checkpoint loaded  (trained for {cfg['num_classes']} classes, "
          f"img_size={cfg['img_size']})")

    backbone = full_model.backbone   # nn.Module with classifier=Identity
    return backbone



#  MODEL  – frozen EfficientNet-B2 backbone + projection + LSTM head


class SignLanguageLSTM(nn.Module):
    def __init__(self, cnn_checkpoint: str, num_classes: int, cfg: dict):
        super().__init__()

        # CNN backbone: load the full ASLClassifier then keep only backbone 
        self.cnn = load_cnn_backbone(cnn_checkpoint)

        # freeze the CNN  only projection + LSTM are trained
        for p in self.cnn.parameters():
            p.requires_grad = False

        # projection  1408 → proj_dim 
        self.projection = nn.Sequential(
            nn.Linear(cfg["cnn_feature_dim"], cfg["proj_dim"], bias=False),
            nn.LayerNorm(cfg["proj_dim"]),
            nn.ReLU(inplace=True),
        )

        # LSTM 
        self.lstm = nn.LSTM(
            input_size    = cfg["proj_dim"],
            hidden_size   = cfg["hidden_size"],
            num_layers    = cfg["num_layers"],
            batch_first   = True,
            dropout       = cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            bidirectional = False,
        )

        # classifier head 
        self.head = nn.Sequential(
            nn.LayerNorm(cfg["hidden_size"]),
            nn.Dropout(p=cfg["dropout"]),
            nn.Linear(cfg["hidden_size"], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg["dropout"] / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.projection.modules()) + list(self.head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # set forget-gate bias = 1 for better gradient flow
                h = param.size(0) // 4
                param.data[h: 2 * h].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C, H, W) a batch of frame sequences
        """
        B, T, C, H, W = x.shape

        # 1. run CNN on every frame ,backbone is frozen, no_grad
        x_flat = x.view(B * T, C, H, W)
        with torch.no_grad():
            feats = self.cnn(x_flat)          # (B*T, 1408)
        feats = feats.view(B, T, -1)          # (B, T, 1408)

        # 2. project embeddings 
        projected = self.projection(feats)    # (B, T, proj_dim)

        # 3. LSTM over time 
        _, (hidden, _) = self.lstm(projected) # hidden: (num_layers, B, hidden)
        last_hidden    = hidden[-1]           # (B, hidden)

        # 4. classify 
        return self.head(last_hidden)         # (B, num_classes)



#  TRAINING UTILITIES


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    criterion    = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    running_loss = correct = total = 0

    for clips, labels in loader:
        clips  = clips.to(DEVICE, non_blocking=True)   # (B, T, C, H, W)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=cfg["use_amp"]):
            logits = model(clips)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * clips.size(0)
        correct      += (logits.argmax(dim=-1) == labels).sum().item()
        total        += clips.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    criterion    = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for clips, labels in loader:
        clips  = clips.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast("cuda", enabled=CFG["use_amp"]):
            logits = model(clips)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * clips.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "loss"   : running_loss / len(all_targets),
        "acc"    : accuracy_score(all_targets, all_preds),
        "preds"  : all_preds,
        "targets": all_targets,
    }


class EarlyStopping:
    def __init__(self, patience: int = 8, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience



#  MAIN TRAINING LOOP


def train(cfg: dict):
    train_loader, test_loader, class_names = build_dataloaders(cfg)

    with open(os.path.join(cfg["output_dir"], "class_map.json"), "w") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, indent=2)

    model = SignLanguageLSTM(
        cnn_checkpoint = cfg["cnn_checkpoint"],
        num_classes    = cfg["num_classes"],
        cfg            = cfg,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable : {trainable:,} / {total:,}  (CNN is frozen)")

    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler  = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["epochs"] * len(train_loader), T_mult=1, eta_min=1e-6,
    )
    scaler     = GradScaler("cuda", enabled=cfg["use_amp"])
    early_stop = EarlyStopping(patience=cfg["patience"])
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining LSTM  ({cfg['epochs']} epochs)")
    print("-" * 70)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg)
        va = evaluate(model, test_loader)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": best_val_acc, "cfg": cfg},
                os.path.join(cfg["output_dir"], "best_lstm_model.pth"),
            )

        print(
            f"  Ep {epoch:02d}/{cfg['epochs']}  |  "
            f"Loss {tr['loss']:.4f}  Acc {tr['acc']:.4f}  |  "
            f"Val Loss {va['loss']:.4f}  Acc {va['acc']:.4f}  |  "
            f"Best {best_val_acc:.4f}  |  {time.time()-t0:.1f}s"
        )

        if early_stop(va["loss"], model):
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)

    return model, test_loader, class_names, history



#  PLOTS & EVALUATION
#

def plot_training_curves(history: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    for ax, k_tr, k_va, title in zip(
        axes,
        ["train_loss", "train_acc"],
        ["val_loss",   "val_acc"],
        ["Loss",       "Accuracy"],
    ):
        ax.plot(epochs, history[k_tr], "b-o", markersize=3, label="Train")
        ax.plot(epochs, history[k_va], "r-o", markersize=3, label="Val")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("LSTM + Hand-Crop  —  Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def full_evaluation(model, test_loader, class_names: list, output_dir: str):
    metrics = evaluate(model, test_loader)
    print(f"\nFinal test results")
    print("-" * 40)
    print(f"  Loss     : {metrics['loss']:.4f}")
    print(f"  Accuracy : {metrics['acc']*100:.2f}%\n")
    print(classification_report(metrics["targets"], metrics["preds"],
                                 target_names=class_names, digits=4))

    cm = confusion_matrix(metrics["targets"], metrics["preds"], normalize="true")
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.4)
    ax.set_title("Confusion Matrix — Dynamic ASL  (hand-cropped)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()



#  ENTRY POINt

if __name__ == "__main__":
    model, test_loader, class_names, history = train(CFG)
    plot_training_curves(history, CFG["output_dir"])
    full_evaluation(model, test_loader, class_names, CFG["output_dir"])