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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

import splitfolders

input_folder = "/kaggle/input/datasets/rishisujitham25b128/signlang/SignAlphaSet"
output_folder = "/kaggle/working/data_split"

# Split with a ratio of 80% training and 20% validation
splitfolders.ratio(input_folder, output=output_folder, 
                   seed=42, ratio=(.8, .2), 
                   group_prefix=None, move=False)

# edit paths, batch size, epochs etc here — nothing else should need changing
CFG = {
    "train_dir"  : "/kaggle/working/data_split/train",
    "test_dir"   : "/kaggle/working/data_split/val",
    "output_dir"      : "/kaggle/working/asl_output",
    "num_classes"     : 36,        # 0-9 + A-Z, gets overridden from actual folder count anyway
    "img_size"        : 224,
    "dropout"         : 0.40,
    "epochs"          : 40,
    "batch_size"      : 128,
    "num_workers"     : 4,
    "lr"              : 3e-4,
    "weight_decay"    : 1e-4,
    "label_smoothing" : 0.10,
    "patience"        : 8,
    "mixup_alpha"     : 0.30,
    "seed"            : 42,
    "use_amp"         : True,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True  # faster once input size is fixed


set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
print(f"[INFO] Device    : {DEVICE}")
print(f"[INFO] GPU count : {torch.cuda.device_count()}")
print(f"[INFO] Output    : {CFG['output_dir']}")


def get_base_model(model):
    # DataParallel wraps the model under .module, need to unwrap to access custom methods
    return model.module if isinstance(model, nn.DataParallel) else model


def build_transforms(img_size, split):
    # imagenet mean/std since we're using a pretrained backbone
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            # resize slightly larger then crop randomly — gives positional variation
            transforms.Resize(
                (img_size + 32, img_size + 32),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomCrop(img_size, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=20,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.10, 0.10),
                scale=(0.85, 1.15),
                shear=10,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.25,
                p=0.30,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            # colour augmentation to handle different lighting conditions
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.06),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # randomly erase a small patch to simulate partial hand occlusion
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ])
    else:
        # no augmentation at test time, just clean resize and normalise
        return transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_weighted_sampler(dataset):
    # if some classes have fewer images, oversample them so the model sees all classes equally
    targets       = np.array(dataset.targets)
    class_counts  = np.bincount(targets, minlength=len(dataset.classes))
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    sample_weights = torch.tensor(class_weights[targets], dtype=torch.float)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(cfg):
    train_ds = datasets.ImageFolder(root=cfg["train_dir"], transform=build_transforms(cfg["img_size"], "train"))
    test_ds  = datasets.ImageFolder(root=cfg["test_dir"],  transform=build_transforms(cfg["img_size"], "test"))

    # override whatever was in CFG with the actual folder count
    cfg["num_classes"] = len(train_ds.classes)

    sampler = get_weighted_sampler(train_ds)

    train_loader = DataLoader(
        dataset            = train_ds,
        batch_size         = cfg["batch_size"],
        sampler            = sampler,
        num_workers        = cfg["num_workers"],
        pin_memory         = True,        # faster CPU->GPU transfer
        drop_last          = True,        # avoid tiny last batch breaking batchnorm
        persistent_workers = (cfg["num_workers"] > 0),
        prefetch_factor    = 2,
    )
    test_loader = DataLoader(
        dataset            = test_ds,
        batch_size         = cfg["batch_size"] * 2,  # no gradients so can fit more
        shuffle            = False,
        num_workers        = cfg["num_workers"],
        pin_memory         = True,
        persistent_workers = (cfg["num_workers"] > 0),
        prefetch_factor    = 2,
    )

    print(f"[INFO] Train   : {len(train_ds):,} images | {len(train_ds.classes)} classes")
    print(f"[INFO] Test    : {len(test_ds):,} images")
    print(f"[INFO] Classes : {train_ds.classes}")
    return train_loader, test_loader, train_ds.classes


class ASLClassifier(nn.Module):
    # EfficientNet-B2 as the backbone with a custom head on top
    # trained in two stages: head only first, then backbone fine-tuning

    def __init__(self, num_classes, dropout=0.40):
        super().__init__()

        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features   = self.backbone.classifier[1].in_features  # 1408 for B2
        self.backbone.classifier = nn.Identity()  # strip original head, we replace it

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features, momentum=0.01, eps=1e-3),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),   # swish activation, same as what efficientnet uses internally
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[INFO] Backbone frozen, training head only")

    def unfreeze_backbone(self, unfreeze_blocks=4):
        # only unfreeze the last few blocks — earlier blocks have generic features we don't need to change
        all_blocks = list(self.backbone.features.children())
        for block in all_blocks[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Unfroze last {unfreeze_blocks} blocks | trainable params: {trainable:,}")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def mixup_batch(images, labels, alpha, num_classes):
    # blend two random images together and mix their labels proportionally
    # forces the model to learn smoother decision boundaries
    if alpha <= 0:
        return images, F.one_hot(labels, num_classes).float()

    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]

    y_a          = F.one_hot(labels,      num_classes).float()
    y_b          = F.one_hot(labels[idx], num_classes).float()
    mixed_labels = lam * y_a + (1.0 - lam) * y_b

    return mixed, mixed_labels


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        images, soft_labels = mixup_batch(images, labels, cfg["mixup_alpha"], cfg["num_classes"])

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg["use_amp"]):
            logits    = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            loss      = -(soft_labels * log_probs).sum(dim=-1).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        hard_labels   = soft_labels.argmax(dim=-1)
        correct      += (logits.argmax(dim=-1) == hard_labels).sum().item()
        total        += images.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(enabled=CFG["use_amp"]):
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
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
        state = get_base_model(model).state_dict()

        if val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in state.items()}
        else:
            self.counter += 1

        return self.counter >= self.patience


def train(cfg):
    train_loader, test_loader, class_names = build_dataloaders(cfg)

    with open(os.path.join(cfg["output_dir"], "class_map.json"), "w") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, indent=2)

    base_model = ASLClassifier(num_classes=cfg["num_classes"], dropout=cfg["dropout"])

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    model = model.to(DEVICE)
    print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    criterion_hard   = nn.CrossEntropyLoss()
    scaler           = GradScaler(enabled=cfg["use_amp"])

    # stage 1: freeze backbone and only train the head for a few epochs
    # this warms up the randomly initialised head before we touch the pretrained weights
    WARMUP_EPOCHS = 3
    get_base_model(model).freeze_backbone()

    optimizer_s1 = AdamW(
        filter(lambda p: p.requires_grad, get_base_model(model).parameters()),
        lr=cfg["lr"] * 5,         # higher lr is fine since only the small head is training
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler_s1 = CosineAnnealingWarmRestarts(
        optimizer_s1, T_0=WARMUP_EPOCHS * len(train_loader), T_mult=1, eta_min=1e-6
    )

    print(f"\nStage 1 — head warmup ({WARMUP_EPOCHS} epochs)")
    print("-" * 64)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer_s1, scheduler_s1, scaler, cfg)
        va = evaluate(model, test_loader, criterion_hard)
        print(
            f"  Ep {epoch:02d}/{WARMUP_EPOCHS} | "
            f"Loss {tr['loss']:.4f}  Acc {tr['acc']:.4f} | "
            f"Val Loss {va['loss']:.4f}  Acc {va['acc']:.4f} | "
            f"⏱ {time.time()-t0:.1f}s"
        )

    # stage 2: unfreeze last few backbone blocks and fine-tune everything
    # backbone gets a much lower lr so we don't destroy the pretrained features
    S2_EPOCHS = cfg["epochs"] - WARMUP_EPOCHS
    get_base_model(model).unfreeze_backbone(unfreeze_blocks=4)

    optimizer_s2 = AdamW(
        [
            {"params": get_base_model(model).backbone.parameters(), "lr": cfg["lr"] / 10},
            {"params": get_base_model(model).head.parameters(),     "lr": cfg["lr"]},
        ],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler_s2 = CosineAnnealingWarmRestarts(
        optimizer_s2, T_0=S2_EPOCHS * len(train_loader), T_mult=1, eta_min=1e-7
    )

    early_stop   = EarlyStopping(patience=cfg["patience"])
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nStage 2 — full fine-tune ({S2_EPOCHS} epochs)")
    print("-" * 64)

    for epoch in range(1, S2_EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer_s2, scheduler_s2, scaler, cfg)
        va = evaluate(model, test_loader, criterion_hard)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            state = get_base_model(model).state_dict()
            torch.save(
                {"epoch": epoch, "model_state": state, "val_acc": best_val_acc, "cfg": cfg},
                os.path.join(cfg["output_dir"], "best_model.pth"),
            )

        print(
            f"  Ep {epoch:02d}/{S2_EPOCHS} | "
            f"Loss {tr['loss']:.4f}  Acc {tr['acc']:.4f} | "
            f"Val Loss {va['loss']:.4f}  Acc {va['acc']:.4f} | "
            f"Best {best_val_acc:.4f} | ⏱ {time.time()-t0:.1f}s"
        )

        if early_stop(va["loss"], model):
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

    if early_stop.best_state:
        get_base_model(model).load_state_dict(early_stop.best_state)

    return model, test_loader, class_names, history


def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, key_tr, key_va, title in zip(
        axes,
        ["train_loss", "train_acc"],
        ["val_loss",   "val_acc"],
        ["Loss",       "Accuracy"],
    ):
        ax.plot(epochs, history[key_tr], "b-o", markersize=3, label="Train")
        ax.plot(epochs, history[key_va], "r-o", markersize=3, label="Val")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("ASL Classifier — Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(targets, preds, class_names, output_dir):
    cm = confusion_matrix(targets, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4,
    )
    ax.set_title("Normalised Confusion Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


def full_evaluation(model, test_loader, class_names, output_dir):
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss())

    print(f"\nFinal test results")
    print("-" * 40)
    print(f"  Loss     : {metrics['loss']:.4f}")
    print(f"  Accuracy : {metrics['acc']*100:.2f}%")
    print()
    print(classification_report(metrics["targets"], metrics["preds"], target_names=class_names, digits=4))
    plot_confusion_matrix(metrics["targets"], metrics["preds"], class_names, output_dir)


class ASLPredictor:
    """Run inference on a single image after training is done.

    Usage:
        predictor = ASLPredictor("/kaggle/working/asl_output/best_model.pth")
        label, confidence = predictor.predict("/path/to/hand.jpg")
        print(f"Predicted: {label}  ({confidence*100:.1f}%)")
    """

    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        cfg  = ckpt["cfg"]

        # load straight into ASLClassifier — no DataParallel needed at inference time
        self.model = ASLClassifier(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.transform = build_transforms(cfg["img_size"], "test")
        self.cfg       = cfg

        map_path = os.path.join(os.path.dirname(checkpoint_path), "class_map.json")
        with open(map_path) as f:
            self.class_map = {int(k): v for k, v in json.load(f).items()}

    @torch.inference_mode()
    def predict(self, image_path):
        from PIL import Image
        img    = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)

        with autocast(enabled=self.cfg["use_amp"]):
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=-1)[0]
        idx   = probs.argmax().item()
        return self.class_map[idx], probs[idx].item()


if __name__ == "__main__":
    model, test_loader, class_names, history = train(CFG)

    plot_training_curves(history, CFG["output_dir"])
    full_evaluation(model, test_loader, class_names, CFG["output_dir"])

    sample      = "/kaggle/input/datasets/rishisujitham25b128/asl-images/asl_processed/train/2/P10_2_1000.jpg"
    predictor   = ASLPredictor(os.path.join(CFG["output_dir"], "best_model.pth"))
    label, conf = predictor.predict(sample)
    print(f"\nPredicted: '{label}'  (confidence: {conf*100:.1f}%)")