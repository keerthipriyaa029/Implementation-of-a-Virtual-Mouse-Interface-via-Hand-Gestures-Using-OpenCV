# type: ignore
import pyautogui
import numpy as np
import time

class GestureMapper:
    def __init__(self, screen_size, smoothing=8):
        """
        Initialize the gesture mapper to convert hand gestures to mouse actions
        
        Args:
            screen_size: Tuple of (screen_width, screen_height)
            smoothing: Smoothing factor for mouse movement (higher = smoother but slower)
        """
        self.screen_width, self.screen_height = screen_size
        self.smoothing = smoothing
        self.prev_x, self.prev_y = 0, 0
        self.mode = "control"  # Modes: "control", "draw", "system"
        
        # Prevent errors from mouse going outside screen
        pyautogui.FAILSAFE = False
        
        # Frame rate tracking for gestures
        self.prev_time = 0
        self.current_time = 0
        
        # States for gesture recognition
        self.scroll_active = False
        self.scroll_start_y = 0
        self.prev_fingers = [0, 0, 0, 0, 0]
        
        # Define pinch distance threshold
        self.pinch_threshold = 40
        
        # System control variables
        self.volume_control_active = False
        self.brightness_control_active = False
        self.last_y_pos = 0
        
        # Mode tracking
        self.last_mode_switch_time = 0
        self.mode_switch_cooldown = 1.0  # seconds

    def map_position(self, index_finger_pos, frame_shape, offset=100, scale=1.5):
        """
        Map the position of the index finger to screen coordinates
        
        Args:
            index_finger_pos: (x, y) position of index finger tip
            frame_shape: (frame_width, frame_height) from webcam
            offset: Border offset for control area
            scale: Control sensitivity scale
        
        Returns:
            (screen_x, screen_y) coordinates
        """
        frame_width, frame_height = frame_shape
        cam_x, cam_y = index_finger_pos
        
        # Convert from webcam coordinates to screen coordinates
        # Apply a border offset to create a control area
        # Mirror x-axis for intuitive movement
        input_x = np.interp(cam_x, 
                          (offset, frame_width - offset), 
                          (self.screen_width, 0))
        input_y = np.interp(cam_y, 
                          (offset, frame_height - offset), 
                          (0, self.screen_height))
        
        # Apply smoothing
        smooth_x = self.prev_x + (input_x - self.prev_x) / self.smoothing
        smooth_y = self.prev_y + (input_y - self.prev_y) / self.smoothing
        
        # Update previous positions
        self.prev_x, self.prev_y = smooth_x, smooth_y
        
        return smooth_x, smooth_y
    
    def check_pinch(self, landmarks, finger1, finger2, threshold=None):
        """
        Check if two fingers are pinching
        
        Args:
            landmarks: List of landmark positions
            finger1: Index of first finger landmark (tip)
            finger2: Index of second finger landmark (tip)
            threshold: Optional custom threshold
            
        Returns:
            Boolean indicating if pinch detected and distance
        """
        if len(landmarks) <= max(finger1, finger2):
            return False, None
            
        # Get coordinates of finger tips
        x1, y1 = landmarks[finger1][1:]
        x2, y2 = landmarks[finger2][1:]
        
        # Calculate distance
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        # Use custom threshold if provided, otherwise use default
        pinch_threshold = threshold if threshold is not None else self.pinch_threshold
        
        # Return True if pinching (distance below threshold)
        return distance < pinch_threshold, distance
    
    def check_three_finger_pinch(self, landmarks, finger1, finger2, finger3):
        """
        Check if three fingers are close together (pinching)
        
        Args:
            landmarks: List of landmark positions
            finger1, finger2, finger3: Indices of finger landmarks (tips)
            
        Returns:
            Boolean indicating if pinch detected
        """
        if len(landmarks) <= max(finger1, finger2, finger3):
            return False
            
        # Check distance between each pair of fingers
        pinch1_2, _ = self.check_pinch(landmarks, finger1, finger2)
        pinch1_3, _ = self.check_pinch(landmarks, finger1, finger3)
        pinch2_3, _ = self.check_pinch(landmarks, finger2, finger3)
        
        # All three fingers must be close to each other
        return pinch1_2 and pinch1_3 and pinch2_3
    
    def is_thumb_up(self, landmarks, fingers):
        """
        Check if thumb is pointing up (👍)
        
        Args:
            landmarks: List of landmark positions
            fingers: List of which fingers are up
            
        Returns:
            Boolean indicating if thumb is up
        """
        if len(landmarks) < 21:  # Need all hand landmarks
            return False
            
        # Thumb must be up, others folded
        if fingers[0] != 1 or fingers[1] != 0 or fingers[2] != 0 or fingers[3] != 0 or fingers[4] != 0:
            return False
            
        # Check thumb orientation by comparing y coordinates
        thumb_tip_y = landmarks[4][2]
        thumb_ip_y = landmarks[3][2]
        
        # Thumb tip should be significantly higher than the IP joint for "thumb up"
        return thumb_tip_y < thumb_ip_y - 30
    
    def is_thumb_down(self, landmarks, fingers):
        """
        Check if thumb is pointing down (👎)
        
        Args:
            landmarks: List of landmark positions
            fingers: List of which fingers are up
            
        Returns:
            Boolean indicating if thumb is down
        """
        if len(landmarks) < 21:  # Need all hand landmarks
            return False
            
        # Thumb must be up, others folded
        if fingers[0] != 1 or fingers[1] != 0 or fingers[2] != 0 or fingers[3] != 0 or fingers[4] != 0:
            return False
            
        # Check thumb orientation by comparing y coordinates
        thumb_tip_y = landmarks[4][2]
        thumb_ip_y = landmarks[3][2]
        
        # Thumb tip should be significantly lower than the IP joint for "thumb down"
        return thumb_tip_y > thumb_ip_y + 30
    
    def is_fist(self, fingers):
        """
        Check if hand is in a fist (all fingers folded)
        
        Args:
            fingers: List of which fingers are up
            
        Returns:
            Boolean indicating if fist detected
        """
        return fingers == [0, 0, 0, 0, 0]
    
    def is_open_palm(self, fingers):
        """
        Check if hand is an open palm (all fingers extended)
        
        Args:
            fingers: List of which fingers are up
            
        Returns:
            Boolean indicating if open palm detected
        """
        return fingers == [1, 1, 1, 1, 1]
    
    def interpret_gestures(self, landmarks, fingers, frame_shape):
        """
        Interpret hand landmarks and finger positions to determine mouse actions
        
        Args:
            landmarks: List of landmark positions [id, x, y]
            fingers: List of which fingers are up [thumb, index, middle, ring, pinky]
            frame_shape: Dimensions of the camera frame (width, height)
        
        Returns:
            Action to perform
        """
        # Calculate current frame rate
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.current_time
        
        frame_width, frame_height = frame_shape
        action = None
        
        # Check if we have index finger position
        if len(landmarks) > 8:
            index_x, index_y = landmarks[8][1:]  # Index fingertip
            
            # MODE SWITCHING
            # ---------------
            # Switch to Draw Mode: Open Palm
            if self.is_open_palm(fingers) and time.time() - self.last_mode_switch_time > self.mode_switch_cooldown:
                self.mode = "draw"
                action = "switch_to_draw"
                self.last_mode_switch_time = time.time()
                
            # Switch to Control Mode: Fist
            elif self.is_fist(fingers) and time.time() - self.last_mode_switch_time > self.mode_switch_cooldown:
                self.mode = "control"
                action = "switch_to_control"
                self.last_mode_switch_time = time.time()
            
            # CONTROL MODE ACTIONS
            # -------------------
            if self.mode == "control":
                # Cursor Movement: Only Index Finger Up
                if fingers == [0, 1, 0, 0, 0]:
                    screen_x, screen_y = self.map_position((index_x, index_y), (frame_width, frame_height))
                    pyautogui.moveTo(screen_x, screen_y)
                    action = "move"
                
                # Left Click: Index + Thumb Pinch
                pinching_thumb_index, _ = self.check_pinch(landmarks, 4, 8)  # 4=thumb tip, 8=index tip
                if pinching_thumb_index and fingers[2] == 0:  # Make sure middle finger is down
                    screen_x, screen_y = self.map_position((index_x, index_y), (frame_width, frame_height))
                    pyautogui.click(screen_x, screen_y)
                    action = "left_click"
                    # Add a small delay to prevent multiple clicks
                    time.sleep(0.3)
                
                # Right Click: Index + Middle + Thumb Touching (three-finger pinch)
                if self.check_three_finger_pinch(landmarks, 4, 8, 12):  # thumb, index, middle
                    screen_x, screen_y = self.map_position((index_x, index_y), (frame_width, frame_height))
                    pyautogui.rightClick(screen_x, screen_y)
                    action = "right_click"
                    # Add a small delay to prevent multiple clicks
                    time.sleep(0.3)
                
                # Scroll Mode: Index + Middle Extended
                if fingers == [0, 1, 1, 0, 0]:
                    # Enter scroll mode or continue scrolling
                    if not self.scroll_active:
                        self.scroll_active = True
                        self.scroll_start_y = index_y
                    else:
                        # Determine scroll direction and amount
                        scroll_amount = (self.scroll_start_y - index_y) / 5
                        if abs(scroll_amount) > 1:
                            pyautogui.scroll(int(scroll_amount))
                            action = "scroll_" + ("up" if scroll_amount > 0 else "down")
                            # Reset start position for continuous scrolling
                            self.scroll_start_y = index_y
                else:
                    self.scroll_active = False
                
                # Volume Up: Thumb Up (👍)
                if self.is_thumb_up(landmarks, fingers):
                    pyautogui.press('volumeup')
                    action = "volume_up"
                    time.sleep(0.2)  # Prevent rapid fire
                
                # Volume Down: Thumb Down (👎)
                if self.is_thumb_down(landmarks, fingers):
                    pyautogui.press('volumedown')
                    action = "volume_down"
                    time.sleep(0.2)  # Prevent rapid fire
                
                # Brightness Control: Index + Middle + Ring Up
                if fingers == [0, 1, 1, 1, 0]:
                    action = "brightness_control"
                    # This would typically launch the system brightness control
                    # For Windows, could use keyboard shortcut Win+A to open action center
                    if not self.brightness_control_active:
                        self.brightness_control_active = True
                        pyautogui.hotkey('win', 'a')
                        time.sleep(0.5)
                else:
                    self.brightness_control_active = False
            
            # DRAW MODE ACTIONS
            # ----------------
            elif self.mode == "draw":
                # Draw with index finger
                if fingers[1] == 1:
                    screen_x, screen_y = self.map_position((index_x, index_y), (frame_width, frame_height))
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.dragTo(screen_x, screen_y, button='left')
                    action = "draw"
                    
                # Stop drawing (switch back to control mode or when in fist position)
                if self.is_fist(fingers):
                    pyautogui.mouseUp()
                    action = "stop_draw"
        
        # Store current fingers state
        self.prev_fingers = fingers.copy()
        
        return action, self.mode
    
    def get_mode(self):
        """
        Get the current control mode
        
        Returns:
            Current mode string
        """
        return self.mode 