import json
import os
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from features import landmarks_to_features
from model import SignBiLSTM_V5


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

# Load best_model_v5.pt if available, fallback to best_model.pt
if (BASE_DIR / "best_model_v5.pt").exists():
    MODEL_PATH = BASE_DIR / "best_model_v5.pt"
else:
    MODEL_PATH = BASE_DIR / "best_model.pt"

LABELS_PATH = BASE_DIR / "labels.json"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ============================================================
# CONFIG (V5 MODEL SPECIFICATION)
# ============================================================

SEQUENCE_LENGTH = 32

CONFIDENCE_THRESHOLD = 0.60
STABLE_CONFIDENCE = 0.65

STABLE_PREDICTIONS_REQUIRED = 3

RELEASE_FRAMES_REQUIRED = 5


# ============================================================
# LOAD SCALER
# ============================================================

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# LOAD LABELS
# ============================================================

if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Labels file not found: {LABELS_PATH}"
    )

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

print("Loaded labels:", LABELS)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL (SignBiLSTM_V5)
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# CREATE MODEL
# ============================================================

model = SignBiLSTM_V5(
    input_size=126,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    fc_hidden_size=64,
    num_classes=len(LABELS)
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print()
print("==============================")
print("V5 MODEL LOAD SUCCESS")
print("==============================")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Input features: 126")
print("Sequence length:", SEQUENCE_LENGTH)
print("Classes:", len(LABELS))
print("Labels:", LABELS)
print("==============================")
print()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="ISL Translator V5 API")

# Allow origins dynamically from environment or default to local + wildcard
cors_env = os.environ.get("CORS_ORIGINS", "")
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://isl-translator-v5.vercel.app",
    "https://isl-translator-v5.vercel.app/",
]
if cors_env:
    allowed_origins.extend([origin.strip() for origin in cors_env.split(",") if origin.strip()])
else:
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": "V5",
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": 126,
        "classes": LABELS,
        "num_classes": len(LABELS)
    }



# ============================================================
# SAFE WEBSOCKET SEND
# ============================================================

async def safe_send(websocket: WebSocket, data: dict) -> bool:
    try:
        await websocket.send_json(data)
        return True
    except Exception:
        return False


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/predict")
async def predict(websocket: WebSocket):
    await websocket.accept()

    print()
    print("==============================")
    print("CLIENT CONNECTED")
    print("==============================")

    # ========================================================
    # STATE
    # ========================================================

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    no_hand_count = 0
    last_prediction_index = None
    stable_prediction_count = 0
    accepted_prediction = None
    prediction_window_count = 0

    try:
        while True:
            # ==================================================
            # RECEIVE JSON MESSAGE
            # ==================================================
            message_text = await websocket.receive_text()

            try:
                payload = json.loads(message_text)
            except json.JSONDecodeError:
                continue

            msg_type = payload.get("type", "")
            hands = payload.get("hands", [])

            # ==================================================
            # NO HAND / HANDS DOWN
            # ==================================================
            if msg_type == "no_hand" or len(hands) == 0:
                no_hand_count += 1

                # Reset buffer immediately when hands are dropped
                if len(sequence) > 0:
                    print("HANDS DOWN → RESET SEQUENCE BUFFER TO 0")
                    sequence.clear()
                    last_prediction_index = None
                    stable_prediction_count = 0
                    accepted_prediction = None
                    prediction_window_count = 0

                # Notify frontend immediately with frames: 0
                if not await safe_send(
                    websocket,
                    {
                        "status": "ready" if no_hand_count >= 2 else "no_hand",
                        "text": "",
                        "confidence": 0,
                        "frames": 0,
                        "required": SEQUENCE_LENGTH
                    }
                ):
                    break

                continue

            # ==================================================
            # HAND FOUND
            # ==================================================
            no_hand_count = 0

            # ==================================================
            # FEATURES
            # ==================================================
            features = landmarks_to_features(hands)

            if features.shape[0] != 126:
                print(
                    f"ERROR: Expected 126 features, got {features.shape[0]}"
                )
                continue

            # ==================================================
            # SCALE
            # ==================================================
            features_scaled = scaler.transform(
                features.reshape(1, -1)
            )[0].astype(np.float32)

            # ==================================================
            # ADD FRAME
            # ==================================================
            sequence.append(features_scaled)

            # ==================================================
            # COLLECTING (< 20 frames)
            # ==================================================
            if len(sequence) < SEQUENCE_LENGTH:
                if not await safe_send(
                    websocket,
                    {
                        "status": "collecting",
                        "text": "",
                        "confidence": 0,
                        "frames": len(sequence),
                        "required": SEQUENCE_LENGTH
                    }
                ):
                    break

                continue

            # ==================================================
            # SLIDING WINDOW (20 frames)
            # ==================================================
            prediction_window_count += 1

            X = np.array(sequence, dtype=np.float32)
            X_tensor = torch.tensor(
                X,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # ==================================================
            # PREDICTION
            # ==================================================
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = probabilities.max(dim=1)

            predicted_index = prediction.item()
            confidence_value = confidence.item()

            if predicted_index >= len(LABELS):
                continue

            predicted_label = LABELS[predicted_index]

            # ==================================================
            # LOW CONFIDENCE
            # ==================================================
            if confidence_value < CONFIDENCE_THRESHOLD:
                if not await safe_send(
                    websocket,
                    {
                        "status": "uncertain",
                        "text": "",
                        "confidence": confidence_value,
                        "frames": SEQUENCE_LENGTH,
                        "required": SEQUENCE_LENGTH
                    }
                ):
                    break

                continue

            # ==================================================
            # STABILIZATION
            # ==================================================
            if predicted_index == last_prediction_index:
                stable_prediction_count += 1
            else:
                last_prediction_index = predicted_index
                stable_prediction_count = 1

            # ==================================================
            # STABLE
            # ==================================================
            if (
                stable_prediction_count >= STABLE_PREDICTIONS_REQUIRED
                and confidence_value >= STABLE_CONFIDENCE
            ):
                # ----------------------------------------------
                # NEW SIGN
                # ----------------------------------------------
                if accepted_prediction != predicted_index:
                    accepted_prediction = predicted_index

                    print()
                    print("================================")
                    print("STABLE PREDICTION:", predicted_label)
                    print("CONFIDENCE:", f"{confidence_value:.2f}")
                    print("================================")
                    print()

                    if not await safe_send(
                        websocket,
                        {
                            "status": "prediction",
                            "text": predicted_label,
                            "confidence": confidence_value,
                            "frames": SEQUENCE_LENGTH,
                            "required": SEQUENCE_LENGTH
                        }
                    ):
                        break

                # ----------------------------------------------
                # Same sign continuing
                # ----------------------------------------------
                else:
                    if not await safe_send(
                        websocket,
                        {
                            "status": "tracking",
                            "text": predicted_label,
                            "confidence": confidence_value,
                            "frames": SEQUENCE_LENGTH,
                            "required": SEQUENCE_LENGTH
                        }
                    ):
                        break

            # ==================================================
            # STILL STABILIZING
            # ==================================================
            else:
                if not await safe_send(
                    websocket,
                    {
                        "status": "stabilizing",
                        "text": "",
                        "confidence": confidence_value,
                        "frames": SEQUENCE_LENGTH,
                        "required": SEQUENCE_LENGTH
                    }
                ):
                    break

    except WebSocketDisconnect:
        print()
        print("CLIENT DISCONNECTED")
        print()

    except Exception as e:
        print()
        print("WebSocket error:", repr(e))
        print()

    finally:
        print("Connection cleanup complete.")