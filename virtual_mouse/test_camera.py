# type: ignore
import cv2

def test_camera():
    # Try to open the camera
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Trying alternative camera indices...")
        
        # Try different camera indices
        for i in range(1, 5):
            print(f"Trying camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Success! Camera opened with index {i}")
                break
            else:
                print(f"Camera index {i} failed.")
        
        if not cap.isOpened():
            print("Could not open any camera. Please check your connections and permissions.")
            return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit the test.")
    
    # Display camera feed
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the frame
        cv2.imshow('Camera Test', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed.")

if __name__ == "__main__":
    test_camera() 