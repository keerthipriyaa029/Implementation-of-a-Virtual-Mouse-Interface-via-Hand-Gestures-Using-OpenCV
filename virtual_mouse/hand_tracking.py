# type: ignore
import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5):
        """
        Initialize the hand detector with MediaPipe
        
        Args:
            mode: Whether to detect static images or video stream
            max_hands: Maximum number of hands to detect
            detection_con: Minimum detection confidence
            track_con: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        # Print mediapipe version for debugging
        print(f"MediaPipe version: {mp.__version__}")
        
        try:
            # Initialize MediaPipe hand solutions
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.detection_con,
                min_tracking_confidence=self.track_con,
                model_complexity=1  # Use higher complexity model for better accuracy
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Define special finger landmark indices
            self.tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky fingertips
            
            # Initialize hand tracking state
            self.results = None
            self.hands_detected = False
            
            print("MediaPipe hand tracking initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            raise
    
    def find_hands(self, img, draw=True):
        """
        Process an image to find hands and optionally draw landmarks
        
        Args:
            img: Image to process (BGR format)
            draw: Whether to draw landmarks on the image
        
        Returns:
            Processed image
        """
        if img is None:
            print("Warning: Received empty image")
            return img
            
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        self.results = self.hands.process(img_rgb)
        self.hands_detected = self.results.multi_hand_landmarks is not None
        
        # Draw hand landmarks if hands are detected
        if self.hands_detected and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add a rectangle around the hand for better visibility
                x_min, y_min = img.shape[1], img.shape[0]
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                
                # Add padding
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(img.shape[1], x_max + padding), min(img.shape[0], y_max + padding)
                
                # Draw rectangle
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
                
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Find the positions of landmarks for a specific hand
        
        Args:
            img: Image to process
            hand_no: Which hand to get positions for (if multiple detected)
            draw: Whether to draw circles at landmark positions
        
        Returns:
            List of landmark positions [id, x, y]
        """
        landmark_list = []
        
        if img is None:
            return landmark_list
            
        img_height, img_width, _ = img.shape
        
        if self.results and self.results.multi_hand_landmarks:
            # Check if the requested hand exists
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                # Extract all landmark positions
                for id, lm in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * img_width), int(lm.y * img_height)
                    landmark_list.append([id, cx, cy])
                    
                    # Draw circles at landmark positions if requested
                    if draw:
                        # Use different colors for fingertips
                        if id in self.tip_ids:
                            cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)  # Blue for fingertips
                        else:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Pink for other landmarks
        
        return landmark_list
    
    def fingers_up(self, landmarks):
        """
        Determine which fingers are up based on landmarks
        
        Args:
            landmarks: List of landmark positions
        
        Returns:
            List of 5 binary values indicating if each finger is up
        """
        fingers = []
        
        # Check if landmarks list contains enough points
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
        
        # Thumb: compare x position of tip with x position of thumb IP
        # Adjusted for both left and right hands
        if landmarks[self.tip_ids[0]][1] < landmarks[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers: compare y position of tip with y position of PIP joint (2nd joint)
        for id in range(1, 5):
            if landmarks[self.tip_ids[id]][2] < landmarks[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def find_distance(self, p1, p2, img=None, draw=True, r=10, t=3):
        """
        Calculate distance between two points and optionally draw
        
        Args:
            p1: First landmark index
            p2: Second landmark index
            img: Image to draw on
            draw: Whether to draw the connection
            r: Radius of circles at points
            t: Thickness of connecting line
        
        Returns:
            Distance between points, drawn image, and midpoint coordinates
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None, img, None
            
        hand = self.results.multi_hand_landmarks[0]
        landmarks = hand.landmark
        
        # Get coordinates of the specified landmarks
        img_h, img_w, _ = img.shape if img is not None else (0, 0, 0)
        x1, y1 = int(landmarks[p1].x * img_w), int(landmarks[p1].y * img_h)
        x2, y2 = int(landmarks[p2].x * img_w), int(landmarks[p2].y * img_h)
        
        # Calculate distance
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        mid_point = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Draw if requested
        if draw and img is not None:
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, mid_point, r, (0, 0, 255), cv2.FILLED)
        
        return dist, img, mid_point
        
    def hands_present(self):
        """
        Check if hands are detected in the current frame
        
        Returns:
            Boolean indicating if hands are detected
        """
        return self.hands_detected 