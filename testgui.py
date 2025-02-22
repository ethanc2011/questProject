import torch
import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSlider, QFileDialog, QSizePolicy, QStyle, QComboBox, QMainWindow, QMessageBox, QLineEdit
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QUrl, QTimer
from ultralytics import YOLO
import sys

# Initialize global variables
ball_positions = []
ball_boxes = []
racket_boxes = []
person_boxes = []
initialized = False
paused = False
MAX_TRAIL_LENGTH = 30
frames = []  # Store frames
line_coordinates = None

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Function to select video file
def select_video_file():
    video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
    return video_path

# Create a QApplication instance
app = QApplication(sys.argv)

# Function to create a simple GUI for play/pause
def create_gui():
    window = QWidget()
    window.setWindowTitle("Video Control")
    
    layout = QVBoxLayout()
    
    play_button = QPushButton("Play")
    pause_button = QPushButton("Pause")
    
    layout.addWidget(play_button)
    layout.addWidget(pause_button)
    
    window.setLayout(layout)
    window.show()
    
    # Connect buttons to functions
    play_button.clicked.connect(lambda: set_play_state(True))
    pause_button.clicked.connect(lambda: set_play_state(False))
    
    return window

def set_play_state(state):
    global is_playing
    is_playing = state

# Function to create the final screen
def create_final_screen(result_text):
    final_window = QWidget()
    final_window.setWindowTitle("Final Screen")
    
    layout = QVBoxLayout()
    
    title_label = QLabel("Final Screen")
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)
    
    result_label = QLabel(result_text)
    result_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(result_label)
    
    close_button = QPushButton("Close")
    close_button.clicked.connect(final_window.close)
    layout.addWidget(close_button)
    
    final_window.setLayout(layout)
    final_window.show()

# Create GUI
create_gui()

# Open the video file using a file dialog
video_path = select_video_file()
if not video_path:
    print("No video file selected. Exiting.")
    exit()

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def mean_square_error(y_actual, y_predicted):
    meansquarederror = 0

    for i in range(len(y_actual)):
        meansquarederror += ((y_actual[i][0] - y_predicted[i][0]) ** 2) / len(y_actual)
        
    return meansquarederror

def fit_quadratic(x_data, y_data):
    """
    Fits a quadratic function to the given data points
    Returns the polynomial coefficients, model, poly_features, and R-squared score
    """
    # Reshape x data for scikit-learn
    X = np.array(x_data).reshape(-1, 1)
    
    # Create polynomial features (degree=2 for quadratic)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    # print("before fit: ", X_poly)
    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y_data)
    # print("after fit: ", X_poly)
    # Get the coefficients (a, b, c) for ax^2 + bx + c
    # coeffs = [model.coef_[2], model.coef_[1], model.intercept_]
    
    # Calculate R-squared score
    r2_score = model.score(X_poly, y_data)
    
    return model, poly_features

def calculateBounce(ball_positions):
    if not ball_positions:
        print("Error: ball_positions is empty")
        return -1
        
    print(f"Number of ball positions: {len(ball_positions)}")
    print(f"Sample of ball positions: {ball_positions[:5]}")  # Show first 5 positions
    
    results = []
    min_points = 3  # minimum number of points required for fitting
    len_data = len(ball_positions)
    best_bounce_index = -1
    best_mse = float('inf')

    # Convert ball_positions to separate x and y arrays
    ball_positions = np.array(ball_positions)
    
    for bounce_idx in range(min_points, len_data - min_points):
        # Extract x and y coordinates separately
        x_before = ball_positions[:bounce_idx+1, 0] 
        y_before = ball_positions[:bounce_idx+1, 1]
        x_after = ball_positions[bounce_idx:, 0]
        y_after = ball_positions[bounce_idx:, 1]

        # Ensure proper shape for sklearn
        x_before = x_before.reshape(-1, 1)
        y_before = y_before.reshape(-1, 1)
        x_after = x_after.reshape(-1, 1)
        y_after = y_after.reshape(-1, 1)

        print(f"At bounce_idx {bounce_idx}:")
        print(f"x_before shape: {x_before.shape}")
        print(f"x_after shape: {x_after.shape}")

        # Ensure there are enough points
        if len(x_before) < min_points or len(x_after) < min_points:
            continue

        try:
            # Fit both segments
            model_before, poly_before = fit_quadratic(x_before, y_before)
            model_after, poly_after = fit_quadratic(x_after, y_after)
            
            # Use the correct variable names here
            y_predicted_before = model_before.predict(poly_before.transform(x_before))
            y_predicted_after = model_after.predict(poly_after.transform(x_after))
            
            y_error_before = mean_square_error(y_before, y_predicted_before)
            y_error_after = mean_square_error(y_after, y_predicted_after)

            meansquarederror = y_error_before + y_error_after

            if meansquarederror < best_mse: 
                best_mse = meansquarederror
                best_bounce_index = bounce_idx
                print(f"New best bounce found at index {bounce_idx} with MSE {meansquarederror}")
        
        except Exception as e:
            print(f"Error at bounce_idx {bounce_idx}: {str(e)}")
            continue

    if best_bounce_index >= 0:
        print(f"Best bounce found at position: {ball_positions[best_bounce_index]}, MSE: {best_mse}")
    else:
        print("No bounce index found")
    return best_bounce_index

def draw_line(frame):
    global points, image_copy
    
    # Use the provided video frame instead of loading from file
    image = frame.copy()
    
    # Initialize image copy and points list
    image_copy = image.copy()
    points = []
    
    # Create window and set mouse callback
    cv2.namedWindow('Draw Line')
    cv2.setMouseCallback('Draw Line', mouse_callback)
    
    while True:
        cv2.imshow('Draw Line', image_copy)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
        
        # Reset when 'r' is pressed
        if cv2.waitKey(1) & 0xFF == ord('r'):
            image_copy = image.copy()
            points = []
    
    cv2.destroyAllWindows()
    return points if len(points) == 2 else None

def inOrOut(point1, point2, bounce_point):
    if bounce_point is None or bounce_point[0] is None or bounce_point[1] is None:
        print("Error: bounce_point is None or contains None values.")
        return False  # Or handle as needed

    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    bouncex = bounce_point[0]
    bouncey = bounce_point[1]

    m = (y1 - y2) / (x1 - x2) if (x1 - x2) != 0 else float('inf')  # Avoid division by zero
    b = y1 - m * x1
    liney = m * bouncex + b

    return bouncey <= liney

def getIntersection(ball_positions, best_bounce_index, frame):
    print("Starting intersection")
    print("ball_positions size: ", len(ball_positions))
    print("best_bounce_index", best_bounce_index)

    # Convert ball_positions to numpy array and separate x and y coordinates
    positions = np.array(ball_positions)

    # Check if positions is empty or if best_bounce_index is out of bounds
    if positions.ndim < 2 or positions.shape[0] <= best_bounce_index:
        print("Error: Invalid positions array or best_bounce_index out of bounds.")
        return None, None, frame  # Return None if no intersection is found

    # Assuming you have logic here to calculate x_intersection and y_intersection
    # For example, if no intersection is found, return default values
    x_intersection = 0  # Default value
    y_intersection = 0  # Default value

    # Your existing logic to calculate intersection should go here
    # If an intersection is found, update x_intersection and y_intersection

    print(f"x_intersection: {x_intersection}, y_intersection: {y_intersection}")

    return x_intersection, y_intersection, frame  # Ensure these are valid values

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if (boxAArea > boxBArea):
        iou = interArea / boxBArea if boxBArea != 0 else 0
    else: 
        iou = interArea / boxAArea if boxAArea != 0 else 0

    return iou

def detectContact(racket_box, ball_box, person_box):
    if racket_box and ball_box:
        iou_ball = calculate_iou(racket_box, ball_box)
        iou_racket = calculate_iou(racket_box, person_box)

        print("IoU_Ball: ", iou_ball)
        print("IoU_Racket: ", iou_racket)
            
        if iou_ball > 0.5 and iou_racket < 0.5:
            print("Contact detected")
            return True
    return False

def mouse_callback(event, x, y, flags, param):
    global points, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked point
        points.append((x, y))
        
        # Draw the point
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)
        
        # If we have two points, draw the line
        if len(points) == 2:
            cv2.line(image_copy, points[0], points[1], (0, 255, 0), 2)
            print(f"Line coordinates: Point 1 {points[0]}, Point 2 {points[1]}")

# Global variable to control play state
is_playing = True

# Process video frame by frame
cv2.namedWindow('Tennis Ball Tracking')  # Create a window for displaying frames
while True:
    if not initialized:
        ret, frame = cap.read()
        if ret:
            line_coordinates = draw_line(frame)  # Store the line coordinates
            if line_coordinates:
                print("Final line coordinates:", line_coordinates)
            initialized = True
        else:
            break

    if is_playing:
        ret, frame = cap.read()
        if not ret:
            break

        # Store a copy of the frame
        frames.append(frame.copy())
        
        # Run object detection using YOLOv8
        results = model(frame, device="mps", verbose=False)

        # Initialize boxes for detected objects
        ball_box = None
        racket_box = None
        person_box = None

        # Process results and draw bounding boxes
        for result in results:  # Loop through results for each frame
            boxes = result.boxes  # Get boxes object for the current frame
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Get box coordinates
                conf = box.conf[0].item()  # Get confidence score
                cls = box.cls[0].item()  # Get class index

                # Check if the detected object is a tennis ball
                if conf > 0.5 and int(cls) == 32:  # Assuming '32' is the tennis ball class
                    center_x = int((xmin + xmax) / 2)
                    center_y = int((ymin + ymax) / 2)
                    
                    ball_box = [xmin, ymin, xmax, ymax]
                    ball_positions.append((center_x, center_y))
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                # Check if the detected object is a tennis racket
                if conf > 0.5 and int(cls) == 38:
                    racket_box = [xmin, ymin, xmax, ymax]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

                # Check if the detected object is a person
                if conf > 0.5 and int(cls) == 0:
                    person_box = [xmin, ymin, xmax, ymax]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        # Check for contact
        if detectContact(racket_box, ball_box, person_box):
            print("Stopping video due to contact.")
            break  # Stop the video playback if contact is detected

        cv2.imshow('Tennis Ball Tracking', frame)  # Display the frame in the created window

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        toggle_pause()

# After the video loop ends
print("Video loop ended")
print(f"Total ball positions collected: {len(ball_positions)}")

# Calculate the best bounce index
best_bounce_index = calculateBounce(ball_positions)

# Ensure that the bounce frame is valid
if frames:
    bounce_frame = frames[-1]
else:
    print("No frames collected. Exiting.")
    exit()

# Get intersection coordinates
intersection_x, intersection_y, final_frame = getIntersection(ball_positions, best_bounce_index, bounce_frame)

# Check if intersection coordinates are valid
if intersection_x is not None and intersection_y is not None:
    print(f"Bounce index calculated: {best_bounce_index}")
    point1, point2 = points[0], points[1]
    bounce = (intersection_x, intersection_y)
    result = inOrOut(point1, point2, bounce)

    # Draw the result on the final frame
    cv2.putText(final_frame, "The ball is in" if result else "The ball is out", 
                (50, frame_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    # Create the final screen with buttons and title
    create_final_screen("The ball is in" if result else "The ball is out")

    # Display the final frame with the regression lines
    cv2.imshow('Final Frame', final_frame)  # Display the final frame
    cv2.waitKey(0)  # Wait for a key press to close the window
else:
    print("No valid intersection found. Exiting.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
app.exec_()  # Start the QApplication event loop