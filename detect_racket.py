import torch
import cv2

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

# Path to the tennis video (replace with your actual video path)
video_path = '/Users/ethancai/Downloads/holger3.mp4'

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Variable to control the pause state
paused = False

# Function to toggle the pause state when any key is pressed
def toggle_pause():
    global paused
    if paused:
        paused = False
    else:
        paused = True

# Process video frame by frame
while True:
    if not paused:  # If not paused, read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection using YOLOv7
        results = model(frame)

        # Loop through detected objects and draw bounding boxes
        for det in results.xyxy[0]:  # Results for the first image
            xmin, ymin, xmax, ymax, conf, cls = det.tolist()

            # Filter out low-confidence detections (confidence > 0.5)
            if conf > 0.5:
                # Draw bounding box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                # Display class label (change if necessary)
                label = f'{model.names[int(cls)]}: {conf:.2f}'
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        cv2.imshow('Tennis Racket Detection', frame)

    # Wait for 1ms, check if any key is pressed
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'q' to exit
    if key == ord('q'):
        break
    # Toggle pause/resume on any other key press
    elif key != 255:  # '255' indicates no key is pressed, so any valid key press will trigger pause
        toggle_pause()

cap.release()
cv2.destroyAllWindows()
