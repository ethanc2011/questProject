import torch
import cv2

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

# Specify the path to the input video file
video_path = '/Users/ethancai/Downloads/Sinner.mp4'  # Replace with your video file path

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Loop over the frames from the video
while cap.isOpened():
    ret, frame = cap.read()

    # Break the loop if no more frames
    if not ret:
        break

    # Run YOLOv7 on the frame
    results = model(frame)

    # Get the detected objects in the frame
    for detection in results.xyxy[0]:  # Each detection is in the format (xmin, ymin, xmax, ymax, confidence, class)
        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6].tolist()

        # Assuming class ID for tennis ball is known or general object detection
        if int(class_id) == 0:  # YOLO typically assigns class 0 to 'person', replace with the correct class_id for tennis ball if fine-tuned
            # Draw bounding box around the detected object (tennis ball)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Add a label for the bounding box
            label = f'Tennis Ball: {confidence:.2f}'
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Tennis Ball Detection', frame)

    # Press 'q' to quit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
cv2.destroyAllWindows()