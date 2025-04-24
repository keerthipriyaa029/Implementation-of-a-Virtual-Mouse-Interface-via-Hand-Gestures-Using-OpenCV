# type: ignore
import numpy as np
import cv2

class KalmanFilter:
    """
    A simple Kalman filter implementation for smoothing hand tracking
    """
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1, dimensions=2):
        """
        Initialize Kalman filter
        
        Args:
            process_variance: How fast the system state can change
            measurement_variance: How noisy the measurements are
            dimensions: Number of dimensions to track (typically 2 for x,y position)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.dimensions = dimensions
        
        # State (position and velocity) - for 2D tracking this is [x, y, vx, vy]
        self.state = np.zeros(2 * dimensions)
        
        # Process covariance - diagonal matrix representing uncertainties in the model
        self.process_covariance = np.eye(2 * dimensions) * self.process_variance
        
        # Initialize transition matrix (state update matrix)
        # [ 1 0 dt 0 ]
        # [ 0 1 0  dt]
        # [ 0 0 1  0 ]
        # [ 0 0 0  1 ]
        self.transition_matrix = np.eye(2 * dimensions)
        for i in range(dimensions):
            self.transition_matrix[i, i + dimensions] = 1.0  # velocity contribution to position
        
        # Measurement matrix - maps state to measurements
        # [ 1 0 0 0 ]
        # [ 0 1 0 0 ]
        self.measurement_matrix = np.zeros((dimensions, 2 * dimensions))
        for i in range(dimensions):
            self.measurement_matrix[i, i] = 1.0
        
        # Measurement covariance - how noisy measurements are
        self.measurement_covariance = np.eye(dimensions) * self.measurement_variance
        
        # State covariance - uncertainty in the state estimation
        self.state_covariance = np.eye(2 * dimensions)
    
    def update(self, measurement):
        """
        Update the Kalman filter with a new measurement
        
        Args:
            measurement: numpy array of measurements [x, y, ...]
            
        Returns:
            Filtered position [x, y, ...]
        """
        # Predict step
        predicted_state = np.dot(self.transition_matrix, self.state)
        predicted_state_cov = (np.dot(np.dot(self.transition_matrix, self.state_covariance), 
                               self.transition_matrix.T) + self.process_covariance)
        
        # Calculate Kalman gain
        innovation_cov = (np.dot(np.dot(self.measurement_matrix, predicted_state_cov), 
                          self.measurement_matrix.T) + self.measurement_covariance)
        kalman_gain = np.dot(np.dot(predicted_state_cov, self.measurement_matrix.T),
                           np.linalg.inv(innovation_cov))
        
        # Update step
        innovation = measurement - np.dot(self.measurement_matrix, predicted_state)
        self.state = predicted_state + np.dot(kalman_gain, innovation)
        
        # Update state covariance
        identity_matrix = np.eye(2 * self.dimensions)
        self.state_covariance = np.dot((identity_matrix - 
                                     np.dot(kalman_gain, self.measurement_matrix)), 
                                     predicted_state_cov)
        
        # Return filtered position
        result = np.dot(self.measurement_matrix, self.state)
        return result

class ExponentialMovingAverage:
    """
    Simple exponential moving average filter
    """
    def __init__(self, alpha=0.5, dimensions=2):
        """
        Initialize EMA filter
        
        Args:
            alpha: Smoothing factor (0-1). Higher values mean less smoothing.
            dimensions: Number of dimensions to filter
        """
        self.alpha = alpha
        self.dimensions = dimensions
        self.last_value = np.zeros(dimensions)
        self.initialized = False
    
    def update(self, measurement):
        """
        Update the filter with a new measurement
        
        Args:
            measurement: New measurement value as numpy array
            
        Returns:
            Filtered measurement
        """
        if not self.initialized:
            self.last_value = measurement
            self.initialized = True
            return measurement
        
        # Calculate EMA
        self.last_value = self.alpha * measurement + (1 - self.alpha) * self.last_value
        return self.last_value

def draw_info_panel(frame, mode, action, fingers):
    """
    Draw information overlay on the frame
    
    Args:
        frame: Video frame to draw on
        mode: Current control mode
        action: Last performed action
        fingers: List of fingers that are up
        
    Returns:
        Frame with info overlay
    """
    # Draw box in top-left corner
    cv2.rectangle(frame, (0, 0), (300, 180), (245, 117, 16), -1)
    
    # Display mode with enhanced description
    mode_text = f"Mode: {mode.capitalize()}"
    cv2.putText(frame, mode_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display more detailed mode description
    if mode == "control":
        mode_desc = "Mouse & system control"
    elif mode == "draw":
        mode_desc = "Drawing on screen"
    else:
        mode_desc = "Unknown mode"
    
    cv2.putText(frame, mode_desc, (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display action with descriptive text
    if action:
        # Format the action string for better display
        action_formatted = action.replace('_', ' ').title()
        action_text = f"Action: {action_formatted}"
    else:
        action_text = "Action: None"
    
    cv2.putText(frame, action_text, (10, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display fingers up (1=up, 0=down) with labels
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    fingers_text = "Fingers: "
    for i, finger in enumerate(fingers):
        fingers_text += "1" if finger else "0"
        if i < 4:
            fingers_text += ","
    
    cv2.putText(frame, fingers_text, (10, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display recognized hand gesture based on fingers state
    gesture_name = "Unknown gesture"
    if fingers == [0, 1, 0, 0, 0]:
        gesture_name = "Index pointing (cursor move)"
    elif fingers == [1, 1, 0, 0, 0] and action == "left_click":
        gesture_name = "Thumb-Index pinch (left click)"
    elif action == "right_click":
        gesture_name = "Three-finger pinch (right click)"
    elif fingers == [0, 1, 1, 0, 0]:
        gesture_name = "Two fingers up (scroll)"
    elif fingers == [1, 1, 1, 1, 1]:
        gesture_name = "Open palm (switch to draw mode)"
    elif fingers == [0, 0, 0, 0, 0]:
        gesture_name = "Fist (switch to control mode)"
    elif action == "volume_up":
        gesture_name = "Thumb up (volume up)"
    elif action == "volume_down":
        gesture_name = "Thumb down (volume down)"
    elif fingers == [0, 1, 1, 1, 0]:
        gesture_name = "Three fingers up (brightness)"
    
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 145), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Helper tip
    cv2.putText(frame, "Press 'q' to quit", (10, 175), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame 