import torch
import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from ultralytics import YOLO

ball_positions = []
paused = False
MAX_TRAIL_LENGTH = 30

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  # Ensure you have the correct YOLOv8 model (adjust the model filename accordingly)

# Path to the tennis video (replace with your actual video path)
video_path = '/Users/ethancai/Downloads/grigor1.mp4'

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Function to toggle the pause state when any key is pressed
def toggle_pause():
    global paused
    paused = not paused

# Process video frame by frame
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection using YOLOv8
        results = model(frame, device = "mps", verbose = False)  # YOLOv8 automatically processes the image

        if len(ball_positions) > 0:  # Only append if it's a new position
            last_pos = ball_positions[-1]
            if (center_x, center_y) != last_pos:  # Avoid duplicate positions
                ball_positions.append((center_x, center_y))
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

        display_positions = ball_positions[-MAX_TRAIL_LENGTH:]
        for i in range(1, len(display_positions)):
            cv2.line(frame, display_positions[i-1], display_positions[i], (0, 255, 0), 2)

        cv2.imshow('Tennis Ball Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        toggle_pause()

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
    
    return model, poly_features, r2_score

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
            model_before, poly_before, _ = fit_quadratic(x_before, y_before)
            model_after, poly_after, _ = fit_quadratic(x_after, y_after)
            
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

# After the video loop ends
print("Video loop ended")
print(f"Total ball positions collected: {len(ball_positions)}")

if len(ball_positions) >= 6:  # Minimum required for analysis
    print("Enough positions collected, calculating bounce...")
    best_bounce_index = calculateBounce(ball_positions)
    print(f"Bounce index calculated: {best_bounce_index}")

    if best_bounce_index > 0:  # Only proceed if a bounce was found
        print("Valid bounce found, creating plot...")
        try:
            # Convert positions to numpy arrays for plotting
            positions_array = np.array(ball_positions)
            print(f"Position array shape: {positions_array.shape}")
            
            # Prepare data for plotting
            ballx_before = positions_array[:best_bounce_index+1, 0]
            bally_before = positions_array[:best_bounce_index+1, 1]
            ballx_after = positions_array[best_bounce_index:, 0]
            bally_after = positions_array[best_bounce_index:, 1]

            print(f"Before bounce points: {len(ballx_before)}")
            print(f"After bounce points: {len(ballx_after)}")

            # Create the plot
            plt.figure(figsize=(12, 6))

            # Plot original points first
            plt.scatter(ballx_before, bally_before, color='blue', alpha=0.5, label='Before Bounce')
            plt.scatter(ballx_after, bally_after, color='red', alpha=0.5, label='After Bounce')

            # Mark the bounce point
            bounce_point = positions_array[best_bounce_index]
            plt.plot(bounce_point[0], bounce_point[1], 'go', markersize=12, label='Bounce Point')

            # Customize the plot
            plt.title('Tennis Ball Trajectory')
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.grid(True)
            plt.legend()

            # Invert y-axis
            plt.gca().invert_yaxis()

            print("Plot created, attempting to show...")
            plt.show(block=True)  # Adding block=True to ensure plot stays visible
            print("Plot should be visible now")

        except Exception as e:
            print(f"Error during plotting: {str(e)}")
    else:
        print("No valid bounce point found")
else:
    print("Not enough ball positions collected for analysis")

print("Program ending...")

# Clean up
cap.release()
cv2.destroyAllWindows()