import torch
import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSlider, QFileDialog, QSizePolicy, QStyle, QComboBox, QMainWindow, QMessageBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QUrl, QTimer
from ultralytics import YOLO
import sys

initialized = False
paused = False
MAX_TRAIL_LENGTH = 30
line_coordinates = None

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  # Ensure you have the correct YOLOv8 model (adjust the model filename accordingly)

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
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    bouncex = bounce_point[0]
    bouncey = bounce_point[1]
    m = (y1-y2)/(x1-x2)
    b = y1-m*x1
    liney = m*bouncex+b
    if(bouncey<=liney):
        return True
    else:
        return False

def getIntersection(ball_positions, best_bounce_index, frame):
    print("starting intersection")
    print("ball_positions size: ", len(ball_positions))
    print("best_bounce_index", best_bounce_index)
    

    # Convert ball_positions to numpy array and separate x and y coordinates
    positions = np.array(ball_positions)
    x_before = positions[:best_bounce_index+1, 0].reshape(-1, 1)
    y_before = positions[:best_bounce_index+1, 1].reshape(-1, 1)
    x_after = positions[best_bounce_index:, 0].reshape(-1, 1)
    y_after = positions[best_bounce_index:, 1].reshape(-1, 1)

    # Fit quadratic functions with optimal bounce point
    model_before, poly_before = fit_quadratic(x_before, y_before)
    model_after, poly_after = fit_quadratic(x_after, y_after)

    # Get quadratic coefficients from the model
    a1 = model_before.coef_[0][2]  # coefficient for x²
    b1 = model_before.coef_[0][1]  # coefficient for x
    c1 = model_before.intercept_[0]  # constant term

    a2 = model_after.coef_[0][2]
    b2 = model_after.coef_[0][1]
    c2 = model_after.intercept_[0]

    # Find intersection by solving a1x² + b1x + c1 = a2x² + b2x + c2
    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    print("starting calculation")
    # Solve quadratic equation
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Choose the x value that lies within our data range
        x_range = [min(min(x_before), min(x_after)), max(max(x_before), max(x_after))]
        if x1 >= x_range[0] and x1 <= x_range[1]:
            x_intersection = x1
        else:
            x_intersection = x2
            
        # Calculate y at intersection point
        y_intersection = a1*x_intersection**2 + b1*x_intersection + c1

        print("x_intersection: ", x_intersection, "y_intersection", y_intersection)
        
        # Generate points for drawing the curves
        # Before bounce: from start to intersection
        x_before_points = np.linspace(min(x_before), x_intersection, 50).reshape(-1, 1)
        y_before_points = model_before.predict(poly_before.transform(x_before_points))
        
        # After bounce: from intersection to end
        x_after_points = np.linspace(x_intersection, max(x_after), 50).reshape(-1, 1)
        y_after_points = model_after.predict(poly_after.transform(x_after_points))
        
        # Convert points to integer coordinates for drawing
        curve_points_before = np.column_stack((x_before_points, y_before_points)).astype(np.int32)
        curve_points_after = np.column_stack((x_after_points, y_after_points)).astype(np.int32)
        
        # Draw the curves
        for i in range(len(curve_points_before) - 1):
            # Draw before-bounce curve in blue
            cv2.line(frame, 
                    tuple(curve_points_before[i]), 
                    tuple(curve_points_before[i + 1]), 
                    (255, 0, 0), 2)  # Blue color
                    
        for i in range(len(curve_points_after) - 1):
            # Draw after-bounce curve in green
            cv2.line(frame, 
                    tuple(curve_points_after[i]), 
                    tuple(curve_points_after[i + 1]), 
                    (0, 255, 0), 2)  # Green color

        # Draw intersection point
        center_coordinates = (int(x_intersection), int(y_intersection))
        radius = 10
        red_color = (0, 0, 255)
        thickness = -1
        
        cv2.circle(frame, center_coordinates, radius, red_color, thickness)
        
        return x_intersection, y_intersection, frame
    
    return None, None  # Return None if no intersection is found

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
            print("contact detected")
            return True

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

# Function to toggle the pause state when any key is pressed
def toggle_pause():
    global paused
    paused = not paused



class VideoDisplay(QWidget):
    def __init__(self, parent=None):
        super(VideoDisplay, self).__init__(parent)
        #self.setFixedSize(frame_width, frame_height)  # Set the size of the video display
        self.layout = QVBoxLayout(self)
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

    def update_frame(self, frame):
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to QImage and display it
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

# Create the main window and layout
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Tennis Ball Tracking")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        # Create video display
        self.video_display = VideoDisplay(self)
        self.layout.addWidget(self.video_display)

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        self.select_video_button = QPushButton("Select Video", self)
        self.select_video_button.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_video_button)

        self.select_points_button = QPushButton("Select Points", self)
        self.select_points_button.clicked.connect(self.select_points)
        self.select_points_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.select_points_button)

        self.process_video_button = QPushButton("Process Video", self)
        self.process_video_button.clicked.connect(self.process_video)
        self.process_video_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.process_video_button)

        self.see_call_button = QPushButton("See The Call", self)
        self.see_call_button.clicked.connect(self.see_call)
        self.see_call_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.see_call_button)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

        # Initialize state variables
        self.line_coordinates = None
        self.frames = []
        self.ball_positions = []
        self.initialized = False
        
    # Process video frame by frame
    def process_frame(self,frame, line_coordinates):
        
        racket_boxes = []
        ball_boxes = []
        person_boxes = []
        # Store a copy of the frame
        self.frames.append(frame.copy())
        
        # Run object detection using YOLOv8
        results = model(frame, device = "mps", verbose = False)

        # Draw the stored line on each frame
        if len(line_coordinates) == 2:
            cv2.line(frame, line_coordinates[0], line_coordinates[1], (0, 255, 0), 2)

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
                    self.ball_positions.append((center_x, center_y))
                    print("detected during video: ", center_x, center_y)
                    ball_boxes = [xmin, ymin, xmax, ymax]

                    # Draw bounding box
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                    # Display class label (change if necessary)
                    label = f'{model.names[int(cls)]}: {conf:.2f}'
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if conf > 0.5 and int(cls) == 38:
                    racket_boxes = [xmin, ymin, xmax, ymax]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Tennis Racket: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if conf > 0.5 and int(cls) == 0:
                    person_boxes = [xmin, ymin, xmax, ymax]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, f'person: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for i in range(1, len(self.ball_positions)):
            cv2.line(frame, self.ball_positions[i-1], self.ball_positions[i], (0, 255, 0), 2)
        
        result = detectContact(racket_boxes, ball_boxes, person_boxes)
        
        return frame, result

    def select_points(self):
        # Logic to select points on the frame
        ret, frame = self.cap.read()
        if ret:
            self.line_coordinates = draw_line(frame)  # Store the line coordinates
            if self.line_coordinates:
                print("Final line coordinates:", self.line_coordinates)
                self.process_video_button.setEnabled(True)  # Enable processing button

    def process_video(self, line_coordinates):
        # Logic to process the video frames
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cur_frame, result = self.process_frame(frame, self.line_coordinates)
            
            if(result == None):
                self.video_display.update_frame(cur_frame)  # Update the video display
                QApplication.processEvents()  # Process events to update the GUI
            else:
                self.see_call_button.setEnabled(True)  # Enable see call button after processing
                break
            
        

    def see_call(self):
        print(self.ball_positions)
        # Logic to display the final frame with regression lines and bounce point
        best_bounce_index = calculateBounce(self.ball_positions)
        bounce_frame = self.frames[-1]
        intersection_x, intersection_y, final_frame = getIntersection(self.ball_positions, best_bounce_index, bounce_frame)
        point1, point2 = self.line_coordinates[0], self.line_coordinates[1]
        bounce = (intersection_x, intersection_y)
        result = inOrOut(point1, point2, bounce)
        cv2.putText(final_frame, "The ball is in" if result else "The ball is out", (50, self.frame_height-100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        
        self.video_display.update_frame(final_frame)
    
    def select_video(self):
        global video_path
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if video_path:
            print(f"Selected video path: {video_path}")

        self.cap = cv2.VideoCapture(video_path)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.select_points_button.setEnabled(True)
        


if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create the application
    main_window = MainWindow()     # Create an instance of MainWindow
    main_window.show()             # Show the main window
    sys.exit(app.exec_())          # Start the event loop