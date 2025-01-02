import os
import numpy as np
import cv2
from gtts import gTTS
import tempfile
import pygame
from ultralytics import YOLO
import threading
import time
import mediapipe as mp
from queue import Queue
import gradio as gr
import sys
import signal

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Exiting Gradio app...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Load the YOLO model and MediaPipe Pose detection
model = YOLO(r"C:\\Users\\KIIT0001\\OneDrive\\Documents\\ml\\trained_yolo11n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants and global variables
KNOWN_HEIGHTS = {
    'person': 170,
    'car': 150,
    'dog': 60,
    'cat': 30,
    'bottle': 30,
    'cupboard': 180,
    'plant': 100,
    'bed': 50
}
FOCAL_LENGTH = 18.75
is_speaking = False
speak_queue = Queue()
camera_matrix = np.array([[3000, 0, 1600], 
                          [0, 3000, 1250], 
                          [0, 0, 1]], dtype=np.float32)

distortion_coefficients = np.array([0.01, 0.2, 0.0001, -0.0001, -0.5], dtype=np.float32)


# EMA filter parameters
alpha = 0.6
previous_depth = 0.0

def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth
    return filtered_depth

def depth_to_distance(depth_value, depth_scale):
    return -1.0 / (depth_value * depth_scale)

def detect_action(pose_landmarks):
    if pose_landmarks is None or not any(p.visibility > 0.5 for p in pose_landmarks.landmark):
        return "stationed"
    
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]

    if left_shoulder.y < right_shoulder.y and left_hip.y > left_shoulder.y:
        return "sitting"
    if left_shoulder.y < right_shoulder.y and left_hip.y < left_shoulder.y:
        return "standing"
    
    shoulder_y_diff = left_shoulder.y - right_shoulder.y
    if shoulder_y_diff > 0.05:
        return "walking"
    if shoulder_y_diff < -0.05:
        return "running"
    
    return "sitting"

def speak_details(details):
    global is_speaking
    if is_speaking:
        speak_queue.put(details)
        return

    is_speaking = True
    
    try:
        print(f"Speaking: {details}")
        tts = gTTS(details, lang='en')
        
        # Create a temporary file for audio playback.
        temp_file_path = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_file_path)
        
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()
        
        # Wait until the audio finishes playing.
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Attempt to delete the temporary file after playback.
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Error deleting temporary file: {e}")

    except Exception as e:
        print(f"Error in speak_details: {e}")
    
    finally:
        is_speaking = False
        
        # Speak the next message if available.
        if not speak_queue.empty():
            next_message = speak_queue.get()
            speak_details(next_message)

def determine_position(x_center, y_center, frame_width, frame_height):
    position = ""
    if x_center < frame_width / 3:
        position += "left "
    elif x_center > 2 * frame_width / 3:
        position += "right "
    
    if y_center < frame_height / 3:
        position += "top"
    elif y_center > frame_height / 3:
        position += "bottom"
    
    return position.strip()

def process_frame(frame, last_speak_time, previous_positions):
    global is_speaking
    current_time = time.time()
    
    results = model(frame)
    
    if results is None or len(results) == 0:
        print("No results detected.")
        return last_speak_time
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    
    frame_height, frame_width, _ = frame.shape
    detected_objects_count = {}
    
    detailed_summary = []  # Initialize detailed_summary here
    
    for result in results:
        for box in result.boxes:
            if box.conf < 0.4:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            action = "stationed"
            
            if object_name == 'person' and pose_results.pose_landmarks:
                action = detect_action(pose_results.pose_landmarks)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            current_position = (center_x, center_y)
            
            if current_position in previous_positions:
                previous_position = previous_positions[current_position]
                distance_moved = np.sqrt((current_position[0] - previous_position[0]) ** 2 + 
                                          (current_position[1] - previous_position[1]) ** 2)
                speed = distance_moved / (current_time - last_speak_time) # Speed in pixels per second
                
                detailed_summary.append(f"Speed: {speed:.2f} px/s")
                previous_positions[current_position] = current_position
            
            if pose_results.pose_landmarks:
                nose_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
                current_depth = nose_landmark.z
                
                filtered_depth = apply_ema_filter(current_depth)
                distance = depth_to_distance(filtered_depth, 1)
            else:
                distance = -1
            
            if distance == -1:
                print(f"Invalid distance for object '{object_name}'. Skipping.")
                continue
            
            position_description = determine_position((x1 + x2) // 2, (y1 + y2) // 2, frame_width, frame_height)
            detected_objects_count[object_name] = detected_objects_count.get(object_name, {"count": 0})
            detected_objects_count[object_name]["count"] += 1
            
            object_details = (
                f"{object_name}: "
                f"Distance {distance:.2f} m, "
                f"Position {position_description}, "
                f"Action: {action}"
            )
            
            detailed_summary.append(object_details) 

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{object_name}: {distance:.2f} m", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    summary_details_list = []
    
    for obj_name, count_info in detected_objects_count.items():
        summary_details_list.append(f"{count_info['count']} {obj_name}")
        
    summary_details_str = ", ".join(summary_details_list)

    detailed_summary_str = "; ".join(detailed_summary)

    print(f"Detected: {summary_details_str}. Details: {detailed_summary_str}.")
    
    if not is_speaking and current_time - last_speak_time > 0.3:
     threading.Thread(
        target=speak_details,
        args=(
            f"Detected objects summary: {summary_details_str}. Details for each: {detailed_summary_str}.",
        ),
    ).start()
    last_speak_time = current_time  # Update after speaking is triggered


    
    return last_speak_time

def start_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_speak_time = time.time()
    previous_positions = {}

    frame_counter = 0
    frame_skip_interval = 2  # Adjust for more frequent processing

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        # Skip frames logic
        if frame_counter % frame_skip_interval != 0:
            frame_counter += 1
            continue

        # Process the frame
        last_speak_time = process_frame(frame, last_speak_time, previous_positions)

        # Increment the frame counter
        frame_counter += 1

        # Optional: Sleep to control loop frequency
        time.sleep(0.01)

        # Show the processed frame in Gradio interface
        yield frame


# Create Gradio interface.
iface= gr.Interface(fn=start_video ,
                    inputs=[] ,
                    outputs=gr.Image(type="numpy") ,
                    live=True)

iface.launch(share=True)
