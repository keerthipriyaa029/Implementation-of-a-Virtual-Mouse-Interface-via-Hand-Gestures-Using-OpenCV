# type: ignore
import cv2
import pyautogui
import numpy as np
import time
import argparse
import os
import sys

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from hand_tracking import HandDetector
from gesture_mapper import GestureMapper
from utils import draw_info_panel, KalmanFilter, ExponentialMovingAverage

def try_camera_indices():
    """Try different camera indices to find a working camera"""
    for i in range(5):  # Try indices 0-4
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Success! Camera opened with index {i}")
                cap.release()
                return i
            cap.release()
        print(f"Camera index {i} failed.")
    return 0  # Default to 0 if no camera found

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gesture-Controlled Virtual Mouse')
    parser.add_argument('--smoothing', type=str, default='ema', 
                        choices=['none', 'ema', 'kalman'],
                        help='Smoothing method to use (none, ema, kalman)')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera index to use (will auto-detect if not specified)')
    parser.add_argument('--smooth-factor', type=float, default=8.0,
                        help='Smoothing factor (higher = smoother but slower)')
    parser.add_argument('--detector-confidence', type=float, default=0.7,
                        help='Hand detector confidence threshold (0.0-1.0)')
    args = parser.parse_args()
    
    # Check OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Auto-detect camera if not specified
    if args.camera is None:
        args.camera = try_camera_indices()
    
    # Initialize webcam with specific parameters
    print(f"Initializing webcam with index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    # Try to set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        print("Try running test_camera.py to find a working camera index.")
        return
    
    # Read test frame to verify camera works
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Error: Could not read frame from camera.")
        print("Try running test_camera.py to troubleshoot.")
        return
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen resolution: {screen_width}x{screen_height}")
    
    # Initialize hand detector
    print(f"Initializing hand detector with confidence threshold {args.detector_confidence}...")
    try:
        detector = HandDetector(detection_con=args.detector_confidence, max_hands=1)
    except Exception as e:
        print(f"Error initializing hand detector: {e}")
        print("Please check that mediapipe is installed correctly.")
        return
    
    # Initialize gesture mapper
    print("Initializing gesture mapper...")
    gesture_mapper = GestureMapper(
        screen_size=(screen_width, screen_height),
        smoothing=args.smooth_factor
    )
    
    # Initialize the appropriate smoothing filter
    if args.smoothing == 'kalman':
        print("Using Kalman filter for smoothing")
        position_filter = KalmanFilter(
            process_variance=1e-5,
            measurement_variance=1e-2
        )
    elif args.smoothing == 'ema':
        print("Using Exponential Moving Average filter for smoothing")
        position_filter = ExponentialMovingAverage(alpha=1/args.smooth_factor)
    else:
        print("No additional smoothing applied")
        position_filter = None
    
    print("Gesture-Controlled Virtual Mouse is running...")
    print("Press 'q' to quit")
    print("\nGesture Guide:")
    print("- Only Index Finger Up: Move cursor")
    print("- Index + Thumb Pinch: Left click")
    print("- Index + Middle + Thumb Pinch: Right click")
    print("- Index + Middle Extended (move up/down): Scroll up/down")
    print("- Open Palm (all 5 fingers): Switch to Draw Mode")
    print("- Fist (all fingers closed): Switch to Control Mode")
    print("- Thumb Up (👍): Volume up")
    print("- Thumb Down (👎): Volume down")
    print("- Index + Middle + Ring Up: Brightness control")
    
    # Main loop
    prev_time = 0
    consecutive_failures = 0
    max_failures = 5
    
    while True:
        try:
            # Read frame from webcam
            success, img = cap.read()
            if not success:
                consecutive_failures += 1
                print(f"Failed to grab frame from camera. Attempt {consecutive_failures}/{max_failures}")
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Exiting.")
                    break
                time.sleep(0.1)
                continue
            consecutive_failures = 0
            
            # Flip the image horizontally for a more intuitive interaction
            img = cv2.flip(img, 1)
            
            # Find hands in the frame
            img = detector.find_hands(img)
            
            # Find landmark positions
            landmarks = detector.find_position(img, draw=True)  # Changed to True to show landmarks
            
            # Determine which fingers are up
            fingers = [0, 0, 0, 0, 0]  # Default to all down
            if landmarks:
                fingers = detector.fingers_up(landmarks)
            
            # Get frame dimensions
            frame_height, frame_width, _ = img.shape
            
            # Interpret gestures and perform corresponding actions
            action, mode = gesture_mapper.interpret_gestures(
                landmarks, 
                fingers, 
                (frame_width, frame_height)
            )
            
            # Apply additional smoothing if enabled
            if position_filter is not None and landmarks and len(landmarks) > 8:
                # Get index finger position
                index_x, index_y = landmarks[8][1:]
                
                # Apply filter
                if isinstance(position_filter, KalmanFilter):
                    filtered_position = position_filter.update(np.array([index_x, index_y]))
                    # Update landmark with filtered position
                    landmarks[8][1] = int(filtered_position[0])
                    landmarks[8][2] = int(filtered_position[1])
                elif isinstance(position_filter, ExponentialMovingAverage):
                    filtered_position = position_filter.update(np.array([index_x, index_y]))
                    # Update landmark with filtered position
                    landmarks[8][1] = int(filtered_position[0])
                    landmarks[8][2] = int(filtered_position[1])
            
            # Draw information panel on the frame
            img = draw_info_panel(img, mode, action, fingers)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            cv2.putText(img, f"FPS: {int(fps)}", (10, frame_height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add camera index display
            cv2.putText(img, f"Camera: {args.camera}", (frame_width - 120, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display the resulting frame
            cv2.imshow("Gesture Mouse", img)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main() 