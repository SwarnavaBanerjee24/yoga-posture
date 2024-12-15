# Real-Time Posture Analysis

This project is a Streamlit-based real-time posture analysis application that uses computer vision techniques to detect and analyze specific yoga and exercise poses. It leverages Mediapipe's pose estimation model and provides feedback on the correctness of poses while tracking the duration a pose is held.

---

## Table of Contents
1. [Features](#features)
2. [Technologies Used]
3. [How It Works]
4. [Setup Instructions]
   - [Prerequisites]
   - [Installation]
   - [Run the Application]
5. [Future Enhancements]
6. [Troubleshooting]

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
Clone the repository:
   ```bash
   git clone https://github.com/SwarnavaBanerjee24/yoga-posture
   ```
### Run the application
   ```bash
   python app.py
   ```

---

### Future Enhancements
- Add more pose classifications and improve pose detection accuracy.
- Incorporate pose-specific improvement suggestions.
- Enable multi-person pose analysis.
- Provide detailed analytics and progress tracking for users.
- Integrate voice feedback for real-time corrections.
- Allow customizable pose thresholds and personalized training routines.
- Develop a mobile app version for broader accessibility.

---

### Troubleshooting
- Camera Access Error: Ensure your webcam is properly connected and not in use by another application.
- Performance Issues: Lower the resolution of the webcam feed or use a device with better processing capabilities.
- Dependency Errors: Verify all required libraries are installed correctly using pip list.
- Streamlit Not Starting: Check if the correct virtual environment is activated and ensure the streamlit package is installed.
