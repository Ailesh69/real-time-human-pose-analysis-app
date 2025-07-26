import cv2
import mediapipe as mp
import numpy as np
import math

# --- Configuration Constants ---
# MediaPipe confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Drawing specifications
POSE_LANDMARK_COLOR = (245, 117, 66) # Orange-ish for pose landmarks
POSE_CONNECTION_COLOR = (245, 66, 230) # Pink-ish for pose connections
HAND_LANDMARK_COLOR = (121, 22, 76)  # Dark purple for hand landmarks
HAND_CONNECTION_COLOR = (121, 44, 250) # Lighter purple for hand connections

# Display box properties
BOX_COLOR = (0, 0, 0) # Black background for info boxes
TEXT_COLOR = (255, 255, 255) # White text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
LINE_HEIGHT = 25
TEXT_THICKNESS = 1

# --- Helper Function for Angle Calculation ---
def calculate_angle(a, b, c):
    

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Clip to avoid floating point errors

    angle = np.degrees(np.arccos(cosine_angle))
    
    if angle > 180.0: # Ensure angle is between 0 and 180 for internal angles
        angle = 360 - angle
    
    return angle

# --- Helper Function for Gesture Detection ---
def detect_gesture(pose_landmarks, hand_landmarks_list, image_width, image_height, mp_pose, mp_hands):
    """
    Detects common gestures based on pose and hand landmarks.
    Includes logic for Arms Crossed, Thumbs Up/Down, Open Palm.
    or "No Gesture" if none is detected.
    """
    gesture = "No Gesture"

    # --- Arms Crossed Detection ---
    if pose_landmarks:
        try:
            r_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            r_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            l_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

            r_wrist_px = np.array([r_wrist.x * image_width, r_wrist.y * image_height])
            l_wrist_px = np.array([l_wrist.x * image_width, l_wrist.y * image_height])
            r_elbow_px = np.array([r_elbow.x * image_width, r_elbow.y * image_height])
            l_elbow_px = np.array([l_elbow.x * image_width, l_elbow.y * image_height])
            r_shoulder_px = np.array([r_shoulder.x * image_width, r_shoulder.y * image_height])
            l_shoulder_px = np.array([l_shoulder.x * image_width, l_shoulder.y * image_height])
            
            # Heuristic: Wrists cross elbows horizontally, and are below shoulders
            if (r_wrist_px[0] < l_elbow_px[0] and l_wrist_px[0] > r_elbow_px[0] and
                r_wrist_px[1] > r_shoulder_px[1] - 30 and l_wrist_px[1] > l_shoulder_px[1] - 30 and
                np.abs(r_wrist_px[0] - l_wrist_px[0]) < 150): # Wrists are horizontally close
                
                return "Arms Crossed"
        except:
            pass # Continue if landmarks are not visible or error in calculation

    # --- Hand Gestures (Thumbs Up/Down, Open Palm) ---
    if hand_landmarks_list:
        for hand_lm in hand_landmarks_list:
            lm_pixels = []
            for lm in hand_lm.landmark:
                lm_pixels.append([lm.x * image_width, lm.y * image_height])
            lm_pixels = np.array(lm_pixels)

            thumb_tip = lm_pixels[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = lm_pixels[mp_hands.HandLandmark.THUMB_IP] # Intermediate Phalange
            thumb_mcp = lm_pixels[mp_hands.HandLandmark.THUMB_MCP]
            
            index_tip = lm_pixels[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = lm_pixels[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_mcp = lm_pixels[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            middle_tip = lm_pixels[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = lm_pixels[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            middle_mcp = lm_pixels[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            ring_tip = lm_pixels[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = lm_pixels[mp_hands.HandLandmark.RING_FINGER_PIP]
            ring_mcp = lm_pixels[mp_hands.HandLandmark.RING_FINGER_MCP]
            
            pinky_tip = lm_pixels[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = lm_pixels[mp_hands.HandLandmark.PINKY_PIP]
            pinky_mcp = lm_pixels[mp_hands.HandLandmark.PINKY_MCP]

            # Helper to check if a finger is curled (angle is small)
            def is_finger_curled(tip, pip, mcp, threshold=90):
                try:
                    return calculate_angle(mcp, pip, tip) < threshold
                except:
                    return False

            # Helper to check if a finger is straight (angle is large)
            def is_finger_straight(tip, pip, mcp, threshold=160):
                try:
                    return calculate_angle(mcp, pip, tip) > threshold
                except:
                    return False

            # --- Thumbs Up ---
            # Thumb straight and pointing up (Y-coord of tip < Y-coord of MCP)
            # All other fingers curled
            if (is_finger_straight(thumb_tip, thumb_ip, thumb_mcp, 150) and
                thumb_tip[1] < thumb_mcp[1] - 50 and # Thumb tip significantly above its base
                is_finger_curled(index_tip, index_pip, index_mcp) and
                is_finger_curled(middle_tip, middle_pip, middle_mcp) and
                is_finger_curled(ring_tip, ring_pip, ring_mcp) and
                is_finger_curled(pinky_tip, pinky_pip, pinky_mcp)):
                
                return "Thumbs Up"

            # --- Thumbs Down ---
            # Thumb straight and pointing down (Y-coord of tip > Y-coord of MCP)
            # All other fingers curled
            if (is_finger_straight(thumb_tip, thumb_ip, thumb_mcp, 150) and
                thumb_tip[1] > thumb_mcp[1] + 50 and # Thumb tip significantly below its base
                is_finger_curled(index_tip, index_pip, index_mcp) and
                is_finger_curled(middle_tip, middle_pip, middle_mcp) and
                is_finger_curled(ring_tip, ring_pip, ring_mcp) and
                is_finger_curled(pinky_tip, pinky_pip, pinky_mcp)):
                
                return "Thumbs Down"
            
            # --- Open Palm / High Five ---
            # All fingers (including thumb) are generally straight/extended
            # And fingers are spread out (distance between thumb and pinky tips)
            if (is_finger_straight(thumb_tip, thumb_ip, thumb_mcp) and
                is_finger_straight(index_tip, index_pip, index_mcp) and
                is_finger_straight(middle_tip, middle_pip, middle_mcp) and
                is_finger_straight(ring_tip, ring_pip, ring_mcp) and
                is_finger_straight(pinky_tip, pinky_pip, pinky_mcp) and
                np.linalg.norm(thumb_tip - pinky_tip) > 150): # Check spread
                
                return "Open Palm / High Five"

    return gesture

# --- Main Program Logic ---
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0) # Camera index 0
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected and not in use by another app.")
        return

    print("Webcam opened successfully. Entering pose estimation loop...")
    
    # --- Initialize Face Detector for Emotion (Placeholder) ---
    face_cascade_path = "data/frontface.xml"
    face_detector_for_emotion = cv2.CascadeClassifier(face_cascade_path)
    if face_detector_for_emotion.empty():
        print(f"Error: Face cascade classifier for emotion detection not loaded from {face_cascade_path}. Emotion detection will be disabled.")
        face_detector_for_emotion = None
    print("Note: Emotion recognition is a placeholder. A full model is not integrated.")

    with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose, \
         mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:

        print("MediaPipe Pose and Hands models loaded. Starting frame processing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Ignoring empty camera frame or stream ended. Attempting next frame...")
                continue

            # Flip for selfie view, convert to RGB for MediaPipe
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            
            image.flags.writeable = False # Make image read-only for MediaPipe processing
            
            # Process image with Pose and Hands models
            pose_results = pose.process(image)
            hand_results = hands.process(image)

            image.flags.writeable = True # Make image writable again for drawing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV display

            # --- EMOTION DETECTION LOGIC (Placeholder) ---
            if face_detector_for_emotion is not None:
                # Use original frame for face detection (can be BGR)
                gray_frame_for_emotion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                faces = face_detector_for_emotion.detectMultiScale(gray_frame_for_emotion, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle for face
                    predicted_emotion = "Neutral" # Placeholder
                    cv2.putText(image, predicted_emotion, (x, y - 10), FONT, 0.9, (0, 0, 255), 2, cv2.LINE_AA) # Red text


            # Draw pose landmarks (skeleton)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=POSE_LANDMARK_COLOR, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=POSE_CONNECTION_COLOR, thickness=2, circle_radius=2)
                )
                
                # --- Angle Calculations & Display Box (Top-Left Corner) ---
                angles_to_display = {}
                try:
                    landmarks = pose_results.pose_landmarks.landmark

                    # Right Arm Angles
                    right_shoulder_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    right_elbow_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_wrist_lm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    right_hip_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                    right_elbow_angle = calculate_angle([right_shoulder_lm.x, right_shoulder_lm.y],
                                                        [right_elbow_lm.x, right_elbow_lm.y],
                                                        [right_wrist_lm.x, right_wrist_lm.y])
                    angles_to_display["Right Elbow"] = int(right_elbow_angle)
                    
                    right_shoulder_angle = calculate_angle([right_hip_lm.x, right_hip_lm.y],
                                                           [right_shoulder_lm.x, right_shoulder_lm.y],
                                                           [right_elbow_lm.x, right_elbow_lm.y])
                    angles_to_display["Right Shoulder"] = int(right_shoulder_angle)

                    # Left Arm Angles
                    left_shoulder_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    left_elbow_lm = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                    left_wrist_lm = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    left_hip_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

                    left_elbow_angle = calculate_angle([left_shoulder_lm.x, left_shoulder_lm.y],
                                                       [left_elbow_lm.x, left_elbow_lm.y],
                                                       [left_wrist_lm.x, left_wrist_lm.y])
                    angles_to_display["Left Elbow"] = int(left_elbow_angle)
                    
                    left_shoulder_angle = calculate_angle([left_hip_lm.x, left_hip_lm.y],
                                                          [left_shoulder_lm.x, left_shoulder_lm.y],
                                                          [left_elbow_lm.x, left_elbow_lm.y])
                    angles_to_display["Left Shoulder"] = int(left_shoulder_angle)
                    
                except Exception as e:
                    # print(f"Error calculating angles: {e}")
                    pass

                # Draw Angle Display Box
                box_start_x_angles, box_start_y_angles = 10, 10
                box_width_angles, box_height_angles = 280, 150
                cv2.rectangle(image, (box_start_x_angles, box_start_y_angles), 
                              (box_start_x_angles + box_width_angles, box_start_y_angles + box_height_angles), 
                              BOX_COLOR, -1)

                y_offset = box_start_y_angles + LINE_HEIGHT
                for label, angle_val in angles_to_display.items():
                    cv2.putText(image, f"{label}: {angle_val} deg",
                                (box_start_x_angles + 10, y_offset), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
                    y_offset += LINE_HEIGHT

            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=HAND_LANDMARK_COLOR, thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=HAND_CONNECTION_COLOR, thickness=2, circle_radius=2)
                    )
            
            # --- Gesture Detection & Display Box (Top-Right Corner) ---
            detected_gesture_text = "No Gesture"
            if pose_results.pose_landmarks or hand_results.multi_hand_landmarks:
                detected_gesture_text = detect_gesture(
                    pose_results.pose_landmarks, 
                    hand_results.multi_hand_landmarks, 
                    image.shape[1], # image width
                    image.shape[0], # image height
                    mp_pose, mp_hands
                )
            
            # Define box properties for gesture display
            box_width_gesture, box_height_gesture = 250, 50
            box_start_x_gesture = image.shape[1] - box_width_gesture - 10
            box_start_y_gesture = 10
            
            cv2.rectangle(image, (box_start_x_gesture, box_start_y_gesture),
                          (box_start_x_gesture + box_width_gesture, box_start_y_gesture + box_height_gesture),
                          BOX_COLOR, -1)

            cv2.putText(image, f"Gesture: {detected_gesture_text}",
                        (box_start_x_gesture + 10, box_start_y_gesture + 30),
                        FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

            cv2.imshow("MediaPipe Pose & Hands & Gestures", image)
            if cv2.waitKey(5) & 0xFF == 27:
                print("ESC key pressed. Exiting loop.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows destroyed. Script finished.")

if __name__ == '__main__':
    main()
    