import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

MODEL_PATH  = Path("lstm_model.pth")
NUM_FRAMES  = 16
CAMERA_IDX  = 0

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'HELLO', 'I', 'J', 'K', 'L', 'M', 'N', 'NO', 'O', 'P', 'Q', 'R', 'S',
           'SORRY', 'T', 'THANKYOU', 'U', 'V', 'W', 'X', 'Y', 'YES', 'Z']



class LSTM_CLASSIFIER(torch.nn.Module):
    def __init__(self, num_class: int, lstm_hidden: int = 512,
                 lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        cnn_features = backbone.features
        adaptive_pool = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()

        projection = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
        )
        self.image_embedding = torch.nn.Sequential(
            cnn_features,
            adaptive_pool,
            flatten,
            projection,
        )

        for param in cnn_features.parameters():
            param.requires_grad = False

        self.lstm = torch.nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        emb = self.image_embedding(x_flat)  # (B*T, 512)
        emb = emb.view(B, T, -1)  # (B, T, 512)

        _, (h_n, _) = self.lstm(emb)  # h_n: (layers, B, H)
        out = self.head(h_n[-1])
        return out


transform = EfficientNet_B0_Weights.DEFAULT.transforms()

def preprocess(bgr_frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(rgb))


def run():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.mps.is_available():
        device = "mps"
    print(f"[INFO] Device: {device}")

    model = LSTM_CLASSIFIER(num_class=len(classes)).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    if not missing_keys and not unexpected_keys:
        print("Model loaded successfully: All keys matched.")
    else:
        print("Model loaded with warnings.")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.eval()
    print(f"[INFO] Loaded model — {len(classes)} classes: {classes}")

    buffer: deque = deque(maxlen=NUM_FRAMES)

    cap = cv2.VideoCapture(CAMERA_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    label     = "collecting frames..."
    prob      = 0.0
    top5      = []

    print("[INFO] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        buffer.append(preprocess(frame))

        if len(buffer) == NUM_FRAMES:
            with torch.inference_mode():
                seq    = torch.stack(list(buffer)).unsqueeze(0).to(device)  # (1,16,C,H,W)
                logits = model(seq)
                probs  = torch.softmax(logits, dim=-1)[0]

            top_p, top_idx = probs.topk(min(5, len(classes)))
            prob   = top_p[0].item()
            label  = classes[top_idx[0].item()]
            top5   = [(classes[i], p.item()) for i, p in zip(top_idx, top_p)]


        H, W = frame.shape[:2] # for background overlay of progress bar
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (W, 70), (15, 15, 15), -1)
        cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

        if len(buffer) < NUM_FRAMES:
            display = f"Buffering... {len(buffer)}/{NUM_FRAMES}"
            cv2.putText(frame, display, (20, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (160, 160, 160), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, label, (20, 52),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 230, 120), 2, cv2.LINE_AA)
            bar_x, bar_y, bar_w, bar_h = 220, 22, 280, 28
            # progress bar
            fill_color = (0, 230, 120) if prob > 0.6 else (0, 200, 220) if prob > 0.35 else (80, 80, 220)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * prob), bar_y + bar_h),
                          fill_color, -1)
            cv2.putText(frame, f"{prob*100:.1f}%", (bar_x + bar_w + 10, bar_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        if top5:
            panel_x, panel_y = 15, H - 185
            cv2.rectangle(frame, (panel_x - 8, panel_y - 22),
                          (panel_x + 300, panel_y + 145),
                          (15, 15, 15), -1)
            cv2.putText(frame, "Top predictions", (panel_x, panel_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (140, 140, 140), 1, cv2.LINE_AA)
            for rank, (cls, p) in enumerate(top5):
                y      = panel_y + 28 + rank * 26
                color  = (0, 230, 120) if rank == 0 else (180, 180, 180)
                weight = 2 if rank == 0 else 1
                cv2.putText(frame, f"{rank+1}. {cls:<12} {p*100:5.1f}%",
                            (panel_x, y), cv2.FONT_HERSHEY_DUPLEX,
                            0.65, color, weight, cv2.LINE_AA)

        cv2.imshow("ASL Inference  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()