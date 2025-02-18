# import torch
# import cv2
# import numpy as np

# # Load YOLOv7 model
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

# # Path to the tennis video (replace with your actual video path)
# video_path = '/Users/ethancai/Downloads/holger3.mp4'

# # Open the video file using OpenCV
# cap = cv2.VideoCapture(video_path)

# # Check if the video file opened successfully
# if not cap.isOpened():
#     print(f"Error: Could not open video file at {video_path}")
#     exit()

# # Variable to control the pause state
# paused = False

# # List to store the ball positions
# ball_positions = []

# # Store the points for perspective transformation
# src_points = []

# # Max number of points for transformation (4 for homography)
# MAX_POINTS = 4

# # Function to toggle the pause state when any key is pressed
# def toggle_pause():
#     global paused
#     paused = not paused

# # Mouse callback to select points and visualize clicks
# def select_points(event, x, y, flags, param):
#     global src_points
#     if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < MAX_POINTS:
#         src_points.append((x, y))
#         print(f"Selected point: {x}, {y}")

# # Set mouse callback for point selection
# cv2.namedWindow('Tennis Ball Tracking')
# cv2.setMouseCallback('Tennis Ball Tracking', select_points)

# # Define the maximum number of positions to display (length of the line)
# MAX_TRAIL_LENGTH = (float('inf'))  # You can adjust this to control the line length

# # Process video frame by frame
# while True:
#     if not paused:  # If not paused, read the next frame
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run object detection using YOLOv7
#         results = model(frame)

#         # Loop through detected objects and draw bounding boxes
#         for det in results.xyxy[0]:  # Results for the first image
#             xmin, ymin, xmax, ymax, conf, cls = det.tolist()

#             # Check if the detected object is a tennis ball (adjust class index if needed)
#             if conf > 0.5 and int(cls) == 32:  # Assuming '32' is the tennis ball class
#                 # Get the ball's center position
#                 center_x = int((xmin + xmax) / 2)
#                 center_y = int((ymin + ymax) / 2)
                
#                 # Append the position to the ball_positions list
#                 ball_positions.append((center_x, center_y))

#                 # Limit the size of the ball_positions list to MAX_TRAIL_LENGTH
#                 if len(ball_positions) > MAX_TRAIL_LENGTH:
#                     ball_positions.pop(0)

#                 # Draw bounding box
#                 cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

#                 # Display class label (change if necessary)
#                 label = f'{model.names[int(cls)]}: {conf:.2f}'
#                 cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Draw the path of the ball (only the last N positions)
#         for i in range(1, len(ball_positions)):
#             # Draw a line between the previous and current positions
#             cv2.line(frame, ball_positions[i-1], ball_positions[i], (0, 255, 0), 2)

#         # Draw selected points on the frame persistently
#         for point in src_points:
#             cv2.circle(frame, point, 8, (0, 0, 255), -1)  # Draw a small red circle for each selected point

#         # If we have selected the source points and destination points
#         if len(src_points) == MAX_POINTS:
#             # Define destination points (these could be hardcoded or interactively selected)
#             dst_points = [(0, 0), (frame.shape[1], 0), (frame.shape[1], frame.shape[0]), (0, frame.shape[0])]

#             # Convert points to NumPy arrays
#             src_pts_np = np.array(src_points, dtype="float32")
#             dst_pts_np = np.array(dst_points, dtype="float32")

#             # Compute the transformation matrix
#             matrix = cv2.getPerspectiveTransform(src_pts_np, dst_pts_np)

#             # Apply the perspective transformation to get the bird's eye view
#             warped_frame = cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

#             # Display the transformed frame
#             cv2.imshow('Bird\'s Eye View', warped_frame)

#         # Display the original frame with tracking and point selection
#         cv2.imshow('Tennis Ball Tracking', frame)

#     # Wait for 1ms, check if any key is pressed
#     key = cv2.waitKey(1) & 0xFF
    
#     # Press 'q' to exit
#     if key == ord('q'):
#         break
#     # Toggle pause/resume on any other key press
#     elif key != 255:
#         toggle_pause()

# cap.release()
# cv2.destroyAllWindows()


import torch
import cv2

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

# Path to the tennis video (replace with your actual video path)
video_path = '/Users/ethancai/Downloads/Sinner.mp4'

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

        # Run object detection using YOLOv7
        results = model(frame)

        # Loop through detected objects and draw bounding boxes
        for det in results.xyxy[0]:  # Results for the first image
            xmin, ymin, xmax, ymax, conf, cls = det.tolist()

            # Check if the detected object is a tennis ball (adjust class index if needed)
            if conf > 0.5 and int(cls) == 32:  # Assuming '0' is the tennis ball class
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