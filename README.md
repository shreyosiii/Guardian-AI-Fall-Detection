# Fall Detection System using Deep Learning

## Overview

This project is an AI-powered Fall Detection System that identifies human fall events from video footage using Deep Learning and Computer Vision techniques. The system can process recorded videos as well as perform live fall detection.

The primary objective is to assist in monitoring elderly people, patients, and individuals at risk by automatically detecting falls and generating alerts.

---

## Features

- Video-based fall detection
- Real-time live detection
- Frame extraction from videos
- Data augmentation for improved model performance
- Deep learning model training and evaluation
- Streamlit-based user interface
- Performance visualization and evaluation

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Matplotlib
- Scikit-learn

---

## Project Structure

```text
.
├── app.py
├── augment_data.py
├── evaluate.py
├── extract_frames.py
├── live_detection.py
├── train_video.py
├── test_video_windows.py
├── fall_video_model.h5
├── streamlit/
├── Fall/
├── No_Fall/
├── frames/
├── videos/
└── README.md
```

---
Note: The trained model file (fall_video_model.h5) is not included due to GitHub file size limitations.
---
## Dataset

The dataset consists of:

- Fall videos
- Non-fall videos
- Extracted image frames

The dataset is used to train a deep learning model capable of distinguishing fall events from normal activities.

---

## Workflow

### 1. Data Collection

Collect fall and non-fall videos.

### 2. Frame Extraction

Extract frames from videos using:

```bash
python extract_frames.py
```

### 3. Data Augmentation

Increase dataset diversity using:

```bash
python augment_data.py
```

### 4. Model Training

Train the fall detection model:

```bash
python train_video.py
```

### 5. Model Evaluation

Evaluate model performance:

```bash
python evaluate.py
```

### 6. Live Detection

Run real-time fall detection:

```bash
python live_detection.py
```

### 7. Streamlit Application

Launch the web interface:

```bash
streamlit run app.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fall-detection-system.git
cd fall-detection-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Example dependencies:

```text
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
streamlit
```

---

## Applications

- Elderly care monitoring
- Hospital patient monitoring
- Smart healthcare systems
- Home safety systems
- Assisted living environments

---

## Future Improvements

- Mobile application integration
- Instant SMS/Email alerts
- Cloud deployment
- Multi-camera support
- Improved model accuracy with larger datasets

---

## Author

Shreyosii

B.Tech Student | Artificial Intelligence & Machine Learning

---

## License

This project is developed for educational and research purposes.
