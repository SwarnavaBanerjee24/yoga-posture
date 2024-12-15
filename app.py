import streamlit as st
import cv2
import mediapipe as mp
import math
from time import time

# Initialize Mediapipe pose model and other variables
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

if 'pose_start_time' not in st.session_state:
    st.session_state.pose_start_time = None
if 'total_hold_time' not in st.session_state:
    st.session_state.total_hold_time = 0
if 'correct_pose' not in st.session_state:
    st.session_state.correct_pose = False
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

st.title("Real-Time Posture Analysis")
st.sidebar.title("Select Target Pose")
pose_options = [
    'Tree Pose',
    'T Pose',
    'Cobra Pose',
    'Warrior II Pose',
    'Downward Dog',
    'Mountain Pose',
    'Chair Pose',
    'Bridge Pose',
    'Plank Pose'
]
target_pose = st.sidebar.selectbox("Choose the pose to analyze:", pose_options)


def detectPose(image, pose):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))
    return image, landmarks


def overlay_angles(image, landmarks):
    # Define font and color (you can customize as per your preferences)
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    light_green = (144, 238, 144)  # light green color (BGR)

    # Calculate angles for key body parts
    left_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    )
    right_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    )
    left_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    )
    right_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    )
    left_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    )
    right_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    )
    left_hip_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    )
    right_hip_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    )

    # Overlay angles directly on the landmarks (as whole numbers)
    cv2.putText(image, str(int(left_elbow_angle)),
                (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0],
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Left Elbow Angle

    cv2.putText(image, str(int(right_elbow_angle)),
                (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0],
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Right Elbow Angle

    cv2.putText(image, str(int(left_shoulder_angle)),
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0],
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Left Shoulder Angle

    cv2.putText(image, str(int(right_shoulder_angle)),
                (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0],
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Right Shoulder Angle

    cv2.putText(image, str(int(left_knee_angle)),
                (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][0],
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Left Knee Angle

    cv2.putText(image, str(int(right_knee_angle)),
                (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][0],
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Right Knee Angle

    cv2.putText(image, str(int(left_hip_angle)),
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Left Hip Angle

    cv2.putText(image, str(int(right_hip_angle)),
                (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1] - 10),
                font, 0.6, light_green, 2, lineType=line_type)  # Right Hip Angle


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def classifyPose(landmarks, target_pose):
    label = 'Unknown Pose'
    color = (0, 0, 255)

    # Calculate angles for key body parts
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    # Pose classification logic
    if target_pose == 'Cobra Pose' and (150 <= left_knee_angle <= 210 and 150 <= right_knee_angle <= 210 and
                                        ((85 <= left_hip_angle <= 125 or 220 <= left_hip_angle <= 260) and
                                         (85 <= right_hip_angle <= 125 or 220 <= right_hip_angle <= 230))):
        label = 'Cobra Pose'

    elif target_pose == 'Warrior II Pose' and (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195 and
                                               80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110 and
                                               (60 < left_knee_angle < 150 or 60 < right_knee_angle < 150)):
        label = 'Warrior II Pose'

    elif target_pose == 'T Pose' and (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195 and
                                      80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110 and
                                      160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
        label = 'T Pose'

    elif target_pose == 'Tree Pose' and ((165 < left_knee_angle < 195 or 165 < right_knee_angle < 195) and
                                         (315 < left_knee_angle < 335 or 25 < right_knee_angle < 45)):
        label = 'Tree Pose'

    elif target_pose == 'Downward Dog' and (45 < left_hip_angle < 80 and 45 < right_hip_angle < 80 and
                                            160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
        label = 'Downward Dog'

    elif target_pose == 'Mountain Pose' and (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195 and
                                             160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
        label = 'Mountain Pose'

    elif target_pose == 'Chair Pose' and ((165 < left_elbow_angle < 195 or 165 < right_elbow_angle < 195) and
                                          (45 < left_knee_angle < 95 or 45 < right_knee_angle < 95)):
        label = 'Chair Pose'

    elif target_pose == 'Bridge Pose' and ((130 < left_hip_angle < 210 or 130 < right_hip_angle < 210) and
                                           (35 < left_shoulder_angle < 75 or 35 < right_shoulder_angle < 75) and
                                           (250 < left_knee_angle < 300 or 250 < right_knee_angle < 300)):
        label = 'Bridge Pose'

    elif target_pose == 'Plank Pose' and (160 < left_elbow_angle < 195 and 160 < right_elbow_angle < 195 and
                                          160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
        label = 'Plank Pose'

    if label == target_pose:
        st.session_state.correct_pose = True
        color = (0, 255, 0)
    else:
        st.session_state.correct_pose = False

    return label, color


def update_timer():
    if st.session_state.correct_pose:
        if st.session_state.pose_start_time is None:
            st.session_state.pose_start_time = time()
        else:
            st.session_state.total_hold_time += time() - st.session_state.pose_start_time
            st.session_state.pose_start_time = time()
    else:
        st.session_state.pose_start_time = None


def main():
    # Create placeholders for pose name and hold time
    pose_display = st.empty()
    hold_time_display = st.empty()

    stframe = st.empty()

    # Check if pose has changed
    if 'last_selected_pose' not in st.session_state:
        st.session_state.last_selected_pose = target_pose
    else:
        if st.session_state.last_selected_pose != target_pose:
            # Pose has changed, reset hold time and start time
            st.session_state.total_hold_time = 0
            st.session_state.pose_start_time = None
            st.session_state.last_selected_pose = target_pose

    # Initialize camera
    if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)

    if not st.session_state.cap.isOpened():
        st.error("Failed to access the camera. Please check your device.")
        return

    while st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break

        frame = cv2.flip(frame, 1)

        frame, landmarks = detectPose(frame, pose)
        if landmarks:
            label, color = classifyPose(landmarks, target_pose)
            overlay_angles(frame, landmarks)
            update_timer()

            pose_color = "green" if st.session_state.correct_pose else "red"

            # Update UI placeholders for pose name and hold time
            pose_display.markdown(f"<h2 style='font-size: 20px; color: {pose_color};'>Pose: {label}</h2>", unsafe_allow_html=True)
            hold_time_display.markdown(f"<h2 style='font-size: 20px;'>Hold Time: {int(st.session_state.total_hold_time)} seconds</h2>",unsafe_allow_html=True)

        # Optionally show the real-time frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    st.session_state.cap.release()


# Run the app
if __name__ == "__main__":
    main()
