import torch
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  # Ensure you have the correct YOLOv8 model (adjust the model filename accordingly)

# Path to the tennis video (replace with your actual video path)
video_path = '/Users/ethancai/Downloads/grigor.mp4'

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Variable to control the pause state
paused = False

# List to store the ball positions
ball_positions = []

# Function to toggle the pause state when any key is pressed
def toggle_pause():
    global paused
    paused = not paused

# Define the maximum number of positions to display (length of the line)
MAX_TRAIL_LENGTH = 3  # You can adjust this to control the line length

# Process video frame by frame
while True:
    if not paused:  # If not paused, read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection using YOLOv8
        results = model(frame, device = "mps", verbose = False)  # YOLOv8 automatically processes the image

        # Loop through detected objects and draw bounding boxes
        for result in results:  # Loop through results for each frame
            boxes = result.boxes  # Get boxes object for the current frame
            for box in boxes:
                # Get box coordinates, confidence, and class
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Get box coordinates
                conf = box.conf[0].item()  # Get confidence score
                cls = box.cls[0].item()  # Get class index

                # Check if the detected object is a tennis ball (adjust class index if needed)
                if conf > 0.5 and int(cls) == 32:  # Assuming '32' is the tennis ball class
                    # Get the ball's center position
                    center_x = int((xmin + xmax) / 2)
                    center_y = int((ymin + ymax) / 2)
                    
                    # Append the position to the ball_positions list
                    ball_positions.append((center_x, center_y))

                    # Limit the size of the ball_positions list to MAX_TRAIL_LENGTH
                    if len(ball_positions) > MAX_TRAIL_LENGTH:
                        ball_positions.pop(0)

                    # Draw bounding box
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                    # Display class label (change if necessary)
                    label = f'{model.names[int(cls)]}: {conf:.2f}'
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw the path of the ball (only the last N positions)
        for i in range(1, len(ball_positions)):
            # Draw a line between the previous and current positions
            cv2.line(frame, ball_positions[i-1], ball_positions[i], (0, 255, 0), 2)
            print(ball_positions[i])
        # Display the frame
        cv2.imshow('Tennis Ball Tracking', frame)

    # Wait for 1ms, check if any key is pressed
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'q' to exit
    if key == ord('q'):
        break
    # Toggle pause/resume on any other key press
    elif key != 255:
        toggle_pause()

cap.release()
cv2.destroyAllWindows()