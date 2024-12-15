# Real-Time Posture Analysis

This project is a Streamlit-based real-time posture analysis application that uses computer vision techniques to detect and analyze specific yoga and exercise poses. It leverages Mediapipe's pose estimation model and provides feedback on the correctness of poses while tracking the duration a pose is held.

---

## Table of Contents
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [How It Works](#how-it-works)
4. [Setup Instructions](#setup-instructions)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Run the Application](#run-the-application)
5. [Code Overview](#code-overview)
6. [Dataset (Optional)](#dataset-optional)
7. [Future Enhancements](#future-enhancements)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Features

- **Real-time webcam-based pose detection**: Uses a live webcam feed to detect body posture.
- **Pose classification**: Recognizes and evaluates 9 yoga and exercise poses:
  - Tree Pose
  - T Pose
  - Cobra Pose
  - Warrior II Pose
  - Downward Dog
  - Mountain Pose
  - Chair Pose
  - Bridge Pose
  - Plank Pose
- **Visual feedback**: Displays calculated joint angles overlaid on the webcam feed.
- **Timer tracking**: Measures how long the correct pose is held.
- **Interactive UI**: Simple Streamlit interface for selecting poses and viewing feedback.

---

## Technologies Used

- **Streamlit**: For the user-friendly web-based interface.
- **OpenCV**: For video capture and image processing.
- **Mediapipe**: For accurate pose estimation and landmark detection.
- **Python**: The primary programming language.

---

## How It Works

1. **Pose Detection**:
   - Mediapipe identifies body landmarks in the webcam feed.
2. **Angle Calculation**:
   - Joint angles (e.g., elbows, shoulders, knees) are calculated.
3. **Pose Classification**:
   - Poses are matched against predefined thresholds for classification.
4. **Feedback and Timer**:
   - Displays whether the user is performing the correct pose and tracks hold duration.

---

## Setup Instructions

### Prerequisites
- Python 3.7 or newer
- A device with an operational webcam

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SwarnavaBanerjee24/yoga-posture
2. Run the application
```bash
python app.py
