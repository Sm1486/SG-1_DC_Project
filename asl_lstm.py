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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

CFG = {
    "data_dir"        : "/kaggle/input/asl-dynamic",
    "cnn_checkpoint"  : "/kaggle/input/your-model/best_model.pth",
    "output_dir"      : "/kaggle/working/lstm_output",
    "num_frames"      : 16,
    "img_size"        : 224,
    "cnn_feature_dim" : 1408,
    "proj_dim"        : 512,
    "hidden_size"     : 256,
    "num_layers"      : 2,
    "dropout"         : 0.40,
    "num_classes"     : 26,
    "epochs"          : 40,
    "batch_size"      : 32,
    "num_workers"     : 4,
    "lr"              : 1e-3,
    "weight_decay"    : 1e-4,
    "label_smoothing" : 0.10,
    "patience"        : 8,
    "seed"            : 42,
    "use_amp"         : True,
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


#-------dataset--------

class DynamicASLDataset(Dataset):
    def __init__(self, data_dir, num_frames, transform):
        self.num_frames = num_frames
        self.transform  = transform
        self.samples    = []
        self.classes    = sorted(os.listdir(data_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for label in self.classes:
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for clip in sorted(os.listdir(label_dir)):
                clip_dir = os.path.join(label_dir, clip)
                if not os.path.isdir(clip_dir):
                    continue
                frames = sorted([
                    os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                if len(frames) >= 2:
                    self.samples.append((frames, self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, frame_paths):
        total = len(frame_paths)
        #evenly spaced indices
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        return [frame_paths[i] for i in indices]

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        selected = self._sample_frames(frame_paths)

        frames = []
        for path in selected:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))

        #stack list of (C,H,W) tensors into (num_frames,C,H,W)
        clip_tensor = torch.stack(frames, dim=0)
        return clip_tensor, label


def build_transforms(img_size, split):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size),
                               interpolation=transforms.InterpolationMode.BILINEAR,
                               antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size),
                               interpolation=transforms.InterpolationMode.BILINEAR,
                               antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_dataloaders(cfg):
    train_dir = os.path.join(cfg["data_dir"], "train")
    test_dir  = os.path.join(cfg["data_dir"], "test")

    if not os.path.isdir(train_dir):
        train_dir = test_dir = cfg["data_dir"]

    train_ds = DynamicASLDataset(train_dir, cfg["num_frames"], build_transforms(cfg["img_size"], "train"))
    test_ds  = DynamicASLDataset(test_dir,  cfg["num_frames"], build_transforms(cfg["img_size"], "test"))

    cfg["num_classes"] = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
        drop_last=True, persistent_workers=(cfg["num_workers"] > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
    )

    print(f"[INFO] Train clips : {len(train_ds):,} | Classes: {train_ds.classes}")
    print(f"[INFO] Test clips  : {len(test_ds):,}")
    return train_loader, test_loader, train_ds.classes


# model

class SignLanguageLSTM(nn.Module):
    def __init__(self, cnn_checkpoint, num_classes, cfg):
        super().__init__()

        #CNN backbone
        backbone = models.efficientnet_b2(weights=None)
        backbone.classifier = nn.Identity()
        self.cnn = backbone

        #load weights from cnn model
        ckpt  = torch.load(cnn_checkpoint, map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}
        self.cnn.load_state_dict(state, strict=False)

        #freeze weights for cnn
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(cfg["cnn_feature_dim"], cfg["proj_dim"], bias=False),
            nn.LayerNorm(cfg["proj_dim"]),
            nn.ReLU(inplace=True),
        )

        #LSTM
        self.lstm = nn.LSTM(
            input_size    = cfg["proj_dim"],
            hidden_size   = cfg["hidden_size"],
            num_layers    = cfg["num_layers"],
            batch_first   = True,
            dropout       = cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            bidirectional = False,
        )

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

        #initalise weights with orthogonal init for better gradient flow
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                hidden_size = param.size(0) // 4
                param.data[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape

        #reshape to (batch*no.frames,c,h,w)
        #pass through cnn
        x = x.view(batch_size * num_frames, C, H, W)

        with torch.no_grad():
            cnn_features = self.cnn(x)

        #reshape back to original shape
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        #project 1408 => 512
        projected = self.projection(cnn_features)

        output, (hidden, cell) = self.lstm(projected)

        last_hidden = hidden[-1]

        #classify
        logits = self.head(last_hidden)
        return logits


#training loop

def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    running_loss = 0.0
    correct = total = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    for clips, labels in loader:
        clips  = clips.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=cfg["use_amp"]):
            logits = model(clips)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        #gradient clipping to prevent vanishing or explosions
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True, foreach=True)
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
    def __init__(self, patience=8, delta=1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


#--------main-----------

def train(cfg):
    train_loader, test_loader, class_names = build_dataloaders(cfg)

    with open(os.path.join(cfg["output_dir"], "class_map.json"), "w") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, indent=2)

    model = SignLanguageLSTM(
        cnn_checkpoint = cfg["cnn_checkpoint"],
        num_classes    = cfg["num_classes"],
        cfg            = cfg,
    ).to(DEVICE)

    #cnn is frozen, only lstm is trained
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params : {trainable:,} / {total:,}")

    #only pass params which require grads to optim
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        betas        = (0.9, 0.999),
        eps          = 1e-8,
        foreach      = True,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["epochs"] * len(train_loader), T_mult=1, eta_min=1e-6
    )
    scaler     = GradScaler("cuda", enabled=cfg["use_amp"])
    early_stop = EarlyStopping(patience=cfg["patience"])
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining LSTM ({cfg['epochs']} epochs)")
    print("-" * 64)

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
            f"  Ep {epoch:02d}/{cfg['epochs']} | "
            f"Loss {tr['loss']:.4f}  Acc {tr['acc']:.4f} | "
            f"Val Loss {va['loss']:.4f}  Acc {va['acc']:.4f} | "
            f"Best {best_val_acc:.4f} | ⏱ {time.time()-t0:.1f}s"
        )

        if early_stop(va["loss"], model):
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)

    return model, test_loader, class_names, history


def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    for ax, key_tr, key_va, title in zip(
        axes, ["train_loss", "train_acc"], ["val_loss", "val_acc"], ["Loss", "Accuracy"]
    ):
        ax.plot(epochs, history[key_tr], "b-o", markersize=3, label="Train")
        ax.plot(epochs, history[key_va], "r-o", markersize=3, label="Val")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("LSTM Classifier — Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def full_evaluation(model, test_loader, class_names, output_dir):
    metrics = evaluate(model, test_loader)
    print(f"\nFinal test results")
    print("-" * 40)
    print(f"  Loss     : {metrics['loss']:.4f}")
    print(f"  Accuracy : {metrics['acc']*100:.2f}%\n")
    print(classification_report(metrics["targets"], metrics["preds"], target_names=class_names, digits=4))

    cm = confusion_matrix(metrics["targets"], metrics["preds"], normalize="true")
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.4)
    ax.set_title("Confusion Matrix — Dynamic ASL", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    model, test_loader, class_names, history = train(CFG)
    plot_training_curves(history, CFG["output_dir"])
    full_evaluation(model, test_loader, class_names, CFG["output_dir"])