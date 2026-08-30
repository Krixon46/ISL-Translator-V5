# ISL Translator

An Indian Sign Language (ISL) translator that uses computer vision and a deep-learning model to recognize selected ISL signs from a live webcam feed and convert them into text.

## Live Demo

**Demo Link:** https://isl-translator-v5.vercel.app/


## Important Notice

**PLEASE WAIT FOR THE APPLICATION TO CONNECT TO THE BACKEND. IT MAY TAKE SOME TIME BEFORE THE CAMERA CONNECTS AND SIGN DETECTION STARTS.**

**THE BACKEND IS CURRENTLY HOSTED ON A FREE RENDER INSTANCE AND MAY BE SLOW, ESPECIALLY WHEN IT IS STARTING UP AFTER A PERIOD OF INACTIVITY.**

**IF THE APPLICATION DOES NOT DETECT SIGNS IMMEDIATELY, WAIT FOR THE BACKEND CONNECTION AND MODEL INITIALIZATION TO COMPLETE BEFORE TRYING AGAIN.**

## Features

* Real-time Indian Sign Language recognition using a webcam
* Hand landmark extraction using MediaPipe
* Deep-learning based temporal sign classification
* Recognition of 9 ISL greeting signs
* 32-frame temporal sequence processing
* 126-feature input representation
* BiLSTM-based sequence modeling
* Temporal attention mechanism
* Confidence score for predictions
* React-based frontend
* Python/FastAPI backend

## Supported Signs

The current model recognizes the following 9 signs:

1. HELLO
2. HOW_ARE_YOU
3. ALRIGHT
4. GOOD_MORNING
5. GOOD_AFTERNOON
6. GOOD_EVENING
7. GOOD_NIGHT
8. THANK_YOU
9. PLEASED

## Machine Learning Model

The current production model is **V5 — SignBiLSTM_V5**.

### Model Configuration

| Parameter                |                   Value |
| ------------------------ | ----------------------: |
| Architecture             |           SignBiLSTM_V5 |
| Input Size               |                     126 |
| Sequence Length          |                      32 |
| Hidden Size              |                     128 |
| LSTM Layers              |                       2 |
| Bidirectional            |                     Yes |
| Dropout                  |                     0.3 |
| FC Hidden Size           |                      64 |
| Number of Classes        |                       9 |
| Hands                    |                       2 |
| Landmarks per Hand       |                      21 |
| Coordinates per Landmark |                       3 |
| Feature Type             | Wrist-relative position |
| Temporal Attention       |                     Yes |

The model processes a sequence of **32 frames**, with **126 features per frame**.

The 126 features correspond to:

`2 hands × 21 landmarks × 3 coordinates = 126 features`

The model uses a bidirectional LSTM followed by temporal attention to learn which frames are most important for identifying a sign.

## Machine Learning Performance

The V5 model achieved:

* **Validation Accuracy: 100%**
* Validation samples: 38
* Correct predictions: 38/38
* Average validation confidence: 0.9700

The exported deployment package was also verified against the original trained model.

The final deployment preprocessing pipeline was tested to ensure that raw landmark data is transformed into the same standardized input used during training.

## Technology Stack

### Frontend

* React
* Vite
* JavaScript
* Web Camera API

### Backend

* Python
* FastAPI
* Uvicorn
* MediaPipe
* PyTorch
* NumPy
* Scikit-learn

### Machine Learning

* BiLSTM
* Temporal Attention
* StandardScaler
* MediaPipe hand landmarks

### Deployment

* Frontend: Vercel
* Backend: Render
* Source Code: GitHub

## System Pipeline

```text
Webcam
   |
   v
Frontend
   |
   v
Video Frames
   |
   v
Backend
   |
   v
MediaPipe Hand Landmark Detection
   |
   v
126 Features per Frame
   |
   v
32-Frame Sequence
   |
   v
StandardScaler
   |
   v
SignBiLSTM_V5
   |
   v
Temporal Attention
   |
   v
9-Class Prediction
   |
   v
Detected ISL Sign
```

## Current Limitations

* The current model supports only 9 signs.
* Recognition requires a sequence of 32 frames.
* Detection performance depends on webcam quality, lighting, hand visibility, and signing style.
* The backend is currently hosted using a free Render instance and can have noticeable startup latency.
* Because the backend is not continuously running on the free hosting tier, the first request may take longer than subsequent requests.
* The current model has been validated primarily on the available dataset and may not perform identically on completely unseen users, backgrounds, camera positions, or signing styles.

## Running Locally

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
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

The frontend and backend configuration should be updated with the appropriate local or production backend URL.

## Deployment

The project uses a separate deployment architecture:

```text
GitHub
  |
  +------------------> Vercel
  |                     |
  |                     v
  |                 React Frontend
  |
  +------------------> Render
                        |
                        v
                    FastAPI Backend
                        |
                        v
                    V5 ML Model
```

The frontend communicates with the deployed FastAPI backend through HTTP/WebSocket requests.

## Model Files

The V5 deployment package contains:

```text
isl_v5_package/
├── best_model_v5.pt
├── labels.json
├── model_config.json
├── scaler.pkl
└── validation_info.json
```

These files contain the trained model, class labels, model configuration, preprocessing scaler, and validation information required for inference.

## Future Improvements

* Add more ISL signs
* Collect more training videos from different users
* Improve performance on unseen users
* Optimize backend inference latency
* Improve WebSocket reliability and responsiveness
* Add sentence-level sign recognition
* Improve continuous sign detection
* Deploy the ML inference service on a faster hosting configuration
* Expand the dataset to cover a larger ISL vocabulary
