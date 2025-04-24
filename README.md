# Implementation-of-a-Virtual-Mouse-Interface-via-Hand-Gestures-Using-OpenCV

The project implements a gesture-controlled virtual mouse using OpenCV and MediaPipe. By tracking hand landmarks in real-time, it enables users to move the cursor, click, and scroll using natural finger gestures. The system enhances human-computer interaction by offering a touchless and intuitive control interface.

# Gesture-Controlled Virtual Mouse

A computer vision application that allows you to control your mouse cursor and system functions using hand gestures captured by your webcam.

## Features

- **Mouse Control**: Move your cursor using your index finger
- **Click Actions**: Perform left-clicks and right-clicks using intuitive pinch gestures
- **Scroll Functionality**: Scroll up and down with two-finger gestures
- **System Controls**: Adjust volume and brightness with dedicated gestures
- **Multiple Modes**:
  - **Control Mode**: Basic mouse operations and system controls
  - **Draw Mode**: Draw on screen by tracking fingertip

## Requirements

- Python 3.7+
- Webcam
- The following Python packages:
  - OpenCV
  - MediaPipe
  - PyAutoGUI
  - NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gesture-mouse.git
   cd gesture-mouse
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Optional command-line arguments:
   ```
   python main.py --smoothing kalman --camera 0 --smooth-factor 8.0
   ```
   - `--smoothing`: Smoothing method to use (`none`, `ema`, or `kalman`)
   - `--camera`: Camera index to use (default is 0)
   - `--smooth-factor`: Smoothing factor (higher = smoother but slower)

3. Press 'q' to quit the application.

## Gesture Guide

| Gesture | Action | Why It Works |
|---------|--------|--------------|
| Only Index Finger Up (others folded) | Move cursor | Easy to hold, stable fingertip for precise control |
| Index + Thumb Pinch | Left click | Feels like a natural "tap" gesture |
| Index + Middle + Thumb Pinch (three-finger pinch) | Right click | Distinct from left click but still comfortable |
| Index + Middle Extended (up/down movement) | Scroll up/down | Like a two-finger swipe, intuitive for touchpad users |
| Open Palm (all five fingers extended) | Switch to Draw Mode | Recognizable, easy to detect mode change |
| Fist (all fingers folded) | Switch to Control Mode | Clear contrast to "open palm" |
| Thumb Up (👍), rest folded | Volume up | Universal gesture, easy to detect |
| Thumb Down (👎), rest folded | Volume down | Intuitive opposite of volume up |
| Index + Middle + Ring Up (three fingers) | Brightness control | Distinct from other gestures, easy to hold |

## Project Structure

- `main.py`: Main script to run the gesture-controlled mouse
- `hand_tracking.py`: Handles MediaPipe hand detection and landmark extraction
- `gesture_mapper.py`: Maps landmarks to gestures and actions
- `utils.py`: Smoothing functions (Kalman filter, EMA) and helper utilities
- `requirements.txt`: List of dependencies

## How It Works

1. **Hand Detection**: The application uses MediaPipe's hand tracking solution to detect hand landmarks in real-time webcam feed.

2. **Gesture Recognition**: Based on the positions of key landmarks (fingertips, joints), the system identifies specific gestures designed to be practical and easy to detect with MediaPipe.

3. **Action Mapping**: Gestures are mapped to mouse actions and system controls using PyAutoGUI.

4. **Smoothing**: Optional smoothing filters (Kalman or Exponential Moving Average) reduce jitter and make cursor movement smooth and natural.

## Troubleshooting

- If the webcam doesn't open, check your camera connection and ensure no other application is using it.
- If hand detection is unreliable, try adjusting lighting conditions or using a plain background.
- If cursor movement is too fast or too slow, adjust the smoothing factor.
- For pinch gestures, make sure your fingers are clearly visible to the camera.
- If system control gestures (volume/brightness) don't work, your system may use different keyboard shortcuts.

## Acknowledgments

- This project uses [MediaPipe](https://mediapipe.dev/) for hand tracking
- [OpenCV](https://opencv.org/) for image processing
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control and system interactions 
