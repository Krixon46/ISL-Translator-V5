# ISL-Translator

A real-time **Indian Sign Language (ISL) Translator** that uses computer vision and deep learning to recognize sign-language gestures from a camera feed and convert them into text.

The system uses **MediaPipe** for hand-landmark extraction and a **BiLSTM with Temporal Attention** trained on ISL gesture sequences for classification.

---

## Live Deployment

**Frontend:** Vercel
**Backend:** Render

> Add your actual deployed URLs here.

* Frontend: `YOUR_VERCEL_URL`
* Backend: `YOUR_RENDER_URL`

---

##  Features

*  Real-time camera-based sign recognition
*  Two-hand landmark tracking
*  BiLSTM-based temporal sequence classification
*  Temporal Attention for focusing on important frames
*  126 features extracted per frame
*  32-frame temporal input sequences
*  Confidence score for predictions
*  WebSocket-based real-time communication
*  Separate frontend and backend deployment
*  Production-ready trained V5 inference model

---

##  System Architecture

```text
                    ┌─────────────────────┐
                    │     User Camera     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     React / Vite    │
                    │      Frontend       │
                    └──────────┬──────────┘
                               │
                         WebSocket
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      MediaPipe      │
                    │   Hand Landmarks    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Extraction  │
                    │ 126 features/frame  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    StandardScaler   │
                    │ Applied exactly once│
                    └──────────┬──────────┘
                               │
                         32 × 126
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SignBiLSTM_V5     │
                    │ + Temporal Attention│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   9 ISL Classes     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Predicted Text +    │
                    │ Confidence           │
                    └─────────────────────┘
```

---

##  Machine Learning Model

The current production model is **V5 — SignBiLSTM_V5**.

### Model Architecture

```text
Input
32 frames × 126 features
        │
        ▼
2-Layer Bidirectional LSTM
Hidden Size = 128
        │
        ▼
256-dimensional temporal representation
        │
        ▼
Temporal Attention
256 → 64 → 1
        │
        ▼
Weighted temporal context
        │
        ▼
Fully Connected Layer
256 → 64
        │
        ▼
ReLU
        │
        ▼
Dropout = 0.3
        │
        ▼
Output Layer
64 → 9
```

### Model Configuration

| Parameter                |                   Value |
| ------------------------ | ----------------------: |
| Model                    |           SignBiLSTM_V5 |
| Input features           |                     126 |
| Sequence length          |                      32 |
| Hidden size              |                     128 |
| LSTM layers              |                       2 |
| Bidirectional            |                     Yes |
| LSTM dropout             |                     0.3 |
| Attention hidden size    |                      64 |
| FC hidden size           |                      64 |
| Number of classes        |                       9 |
| Hands                    |                       2 |
| Landmarks per hand       |                      21 |
| Coordinates per landmark |                       3 |
| Feature representation   | Wrist-relative position |

---

##  Feature Extraction

Each video frame is processed using MediaPipe.

For each detected hand:

```text
21 landmarks
×
3 coordinates (X, Y, Z)
=
63 features
```

For two hands:

```text
63 × 2 = 126 features/frame
```

The model therefore receives:

```text
32 frames × 126 features
```

For inference, the final PyTorch tensor is:

```text
(1, 32, 126)
```

where `1` represents the batch dimension.

---

##  Inference Pipeline

The production inference pipeline is:

```text
Camera Frame
     ↓
MediaPipe Hand Detection
     ↓
21 landmarks per hand
     ↓
Wrist-relative feature extraction
     ↓
126 features/frame
     ↓
Collect 32 frames
     ↓
32 × 126 sequence
     ↓
V5 StandardScaler
     ↓
PyTorch Tensor
     ↓
(1, 32, 126)
     ↓
SignBiLSTM_V5
     ↓
Temporal Attention
     ↓
9-class prediction
     ↓
Label + Confidence
```

### Important

The V5 `StandardScaler` is applied **exactly once** to the raw feature sequence.

Double-scaling the already standardized input must be avoided because it changes the distribution expected by the trained model.

---

##  Supported Signs

The current model recognizes 9 ISL greeting/sign classes:

```text
HELLO
HOW_ARE_YOU
ALRIGHT
GOOD_MORNING
GOOD_AFTERNOON
GOOD_EVENING
GOOD_NIGHT
THANK_YOU
PLEASED
```

The label indices are fixed:

| Index | Label          |
| ----: | -------------- |
|     0 | HELLO          |
|     1 | HOW_ARE_YOU    |
|     2 | ALRIGHT        |
|     3 | GOOD_MORNING   |
|     4 | GOOD_AFTERNOON |
|     5 | GOOD_EVENING   |
|     6 | GOOD_NIGHT     |
|     7 | THANK_YOU      |
|     8 | PLEASED        |

---

##  Model Validation

The V5 model achieved the following result on its held-out validation set:

```text
Validation samples: 38
Correct predictions: 38
Validation accuracy: 100%
```

The model was also checked after exporting the deployment package.

The exported model, scaler, labels, and configuration were verified against the original trained model.

The deployment preprocessing pipeline was additionally verified to reproduce the standardized model input.

> **Note:** The 100% validation accuracy is based on the available validation dataset and should not be interpreted as 100% real-world recognition accuracy. Performance on unseen users, cameras, backgrounds, lighting conditions, signing styles, and genuinely unseen videos may differ.

---

##  Tech Stack

### Frontend

* React
* Vite
* JavaScript
* WebSocket

### Backend

* Python
* FastAPI
* Uvicorn
* WebSocket
* PyTorch
* MediaPipe
* NumPy
* scikit-learn

### Machine Learning

* BiLSTM
* Temporal Attention
* StandardScaler
* Sequence-based classification

### Deployment

* GitHub
* Vercel — Frontend
* Render — Backend

---

##  Project Structure

```text
ISL-Translator/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── backend/
│   ├── assets/
│   │   └── hand_landmarker.task
│   │
│   ├── model/
│   │   └── ...
│   │
│   ├── best_model.pt
│   ├── labels.json
│   ├── model_config.json
│   ├── scaler.pkl
│   ├── server.py
│   └── ...
│
├── README.md
└── ...
```

> The exact directory structure may vary depending on the deployment configuration.

---

##  Local Development

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ISL-Translator
```

### 2. Backend Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn server:app --reload
```

The backend will normally run at:

```text
http://localhost:8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will normally run at:

```text
http://localhost:5173
```

Configure the frontend environment variable to point to the backend.

Example:

```env
VITE_API_URL=http://localhost:8000
```

For production, use the deployed Render backend URL instead.

---

##  Deployment

The application uses a split deployment architecture.

```text
GitHub
   │
   ├──────────────► Vercel
   │                 Frontend
   │
   └──────────────► Render
                     Backend
```

### Frontend — Vercel

The React/Vite application is deployed on Vercel.

The frontend communicates with the backend using the configured production backend URL.

### Backend — Render

The FastAPI application is deployed on Render.

The backend contains the trained V5 model and performs:

* MediaPipe processing
* Feature extraction
* Sequence construction
* Feature scaling
* V5 inference
* WebSocket communication

---

## 🔌 Real-Time Communication

The application uses WebSockets for real-time communication between the frontend and backend.

The general flow is:

```text
Browser Camera
      ↓
Video Frames
      ↓
WebSocket
      ↓
FastAPI Backend
      ↓
Feature Extraction
      ↓
32-frame Sequence
      ↓
V5 Inference
      ↓
Prediction
      ↓
WebSocket
      ↓
Frontend
```

For HTTPS production deployments, WebSocket communication uses secure WebSockets:

```text
wss://
```

---

##  Project Goal

The goal of this project is to build a practical real-time Indian Sign Language translation system that can:

1. Capture signs using a standard camera.
2. Extract meaningful hand landmarks.
3. Understand temporal movement rather than individual frames.
4. Classify complete sign sequences.
5. Return understandable text in real time.

The current implementation focuses on a limited vocabulary of ISL greeting/sign phrases and provides a foundation for expanding the system to a much larger vocabulary.

---

##  Future Improvements

Potential future improvements include:

* Expanding the ISL vocabulary
* Collecting more videos from multiple signers
* Testing on completely unseen users
* Improving robustness to different lighting and backgrounds
* Better continuous-sign segmentation
* Sentence-level translation
* Language-model-based sentence correction
* More advanced temporal architectures
* Larger and more diverse training datasets
* Mobile deployment
* Edge/on-device inference

---

##  Current Limitations

The current model recognizes only the 9 trained classes.

It should not be treated as a complete ISL translation system.

Real-world performance can be affected by:

* Lighting
* Camera angle
* Hand occlusion
* Background
* Distance from camera
* Signing speed
* Different signing styles
* Differences between training and unseen users

The model should therefore be evaluated on genuinely unseen recordings before making claims about real-world accuracy.

---

##  Author

**Krixon**

Computer Science & Engineering

---

##  License



if you decide to release the project under MIT.
