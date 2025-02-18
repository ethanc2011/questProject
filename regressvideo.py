import torch
import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from ultralytics import YOLO

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

# Variable to control the pause state
paused = False

# List to store the ball positions
ball_positions = []

# Function to toggle the pause state when any key is pressed
def toggle_pause():
    global paused
    paused = not paused

# Process video frame by frame
while True:
    if not paused:  # If not paused, read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection using YOLOv8
        results = model(frame, device = "mps", verbose = False)  # YOLOv8 automatically processes the image

        if len(ball_positions) > 0:
            last_pos = ball_positions[-1]
            if (center_x, center_y) != last_pos:
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

# def calculateBounce(ball_positions):
#     results = []
#     min_points = 3  # minimum number of points required for fitting
#     len_data = len(ball_positions)
#     best_bounce_index = -1
#     best_mse = (float('inf'))  # We want to maximize the sum of R² scores

#     for bounce_idx in range(min_points, len_data - min_points): 
#         # Split data
#         # x_before = np.array(x_noisy[:bounce_idx+1]).reshape(1, -1)
#         # y_before = np.array(y_noisy[:bounce_idx+1]).reshape(1, -1)
#         # x_after = np.array(x_noisy[bounce_idx:]).reshape(1, -1)
#         # y_after = np.array(y_noisy[bounce_idx:]).reshape(1, -1)
#         # x_before = (x_noisy[:bounce_idx+1]).reshape(1, -1)
#         # y_before = (y_noisy[:bounce_idx+1]).reshape(1, -1)
#         # x_after = (x_noisy[bounce_idx:]).reshape(1, -1)
#         # y_after = (y_noisy[bounce_idx:]).reshape(1, -1)
#         # ballx_before = np.array(ball_positions[:bounce_idx+1]).reshape(-1, 1)
#         # bally_before = np.array(ball_positions[:bounce_idx+1]).reshape(-1, 1)
#         # ballx_after = np.array(ball_positions[bounce_idx:]).reshape(-1, 1)
#         # bally_after = np.array(ball_positions[bounce_idx:]).reshape(-1, 1)
#         # ballx_before = np.array(ball_positions[:bounce_idx+1]).reshape(1, -1)
#         # bally_before = np.array(ball_positions[:bounce_idx+1]).reshape(1, -1)
#         # ballx_after = np.array(ball_positions[bounce_idx:]).reshape(1, -1)
#         # bally_after = np.array(ball_positions[bounce_idx:]).reshape(1, -1)
#         ballx_before = ([ball_positions[:bounce_idx+1]])
#         bally_before = ([ball_positions[:bounce_idx+1]])
#         ballx_after = ([ball_positions[bounce_idx:]])
#         bally_after = ([ball_positions[bounce_idx:]])
#         print("Ball positions:", ball_positions)
#         print("Ballx Before:", ballx_before)
#         print("Ballx After:", ballx_after)
#         # Ensure there are enough points
#         if len(ballx_before) < min_points or len(ballx_after) < min_points:
#             continue

#         # Fit both segments
#         model_before, poly_before, _ = fit_quadratic(ballx_before, bally_before)
#         model_after, poly_after, _ = fit_quadratic(ballx_after, bally_after)

#         y_predicted_before = model_before.predict(poly_before.transform(ballx_before))
#         y_predicted_after = model_after.predict(poly_after.transform(ballx_after))

#         y_error_before = mean_square_error(bally_before, y_predicted_before)
#         y_error_after = mean_square_error(bally_after, y_predicted_after)

#         meansquarederror = y_error_before + y_error_after

#         if meansquarederror < best_mse: 
#             best_mse = meansquarederror
#             best_bounce_index = bounce_idx
#             print("found")

#     if best_bounce_index >= 0:
#         print(f"best_bounce_x: {ball_positions[best_bounce_index]}, best_bounce_y: {ball_positions[best_bounce_index]}, best_mean_squared_error: {best_mse}")
#     else:
#         print("no bounce index")
#     return best_bounce_index

def calculateBounce(ball_positions):
    if not ball_positions:
        print("Error: ball_positions is empty")
        return -1
        
    print(f"Number of ball positions: {len(ball_positions)}")
    print(f"Sample of ball positions: {ball_positions[:5]}")  # Show first 5 positions
    
    results = []
    min_points = 1  # minimum number of points required for fitting
    len_data = len(ball_positions)
    best_bounce_index = -1
    best_mse = float('inf')

    # Convert ball_positions to separate x and y arrays
    ball_positions = np.array(ball_positions)
    
    for bounce_idx in range(min_points, len_data - min_points):
        print("for loop start")
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
            #print("ballx_before: ", x_before, "bally_before: ", y_before, "ballx_after:c", x_after, "bally_after: ", y_after)
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
    
    return best_bounce_index, x_before, x_after, y_before, y_after

best_bounce_index, x_before, x_after, y_before, y_after = calculateBounce(ball_positions)

if len(ball_positions) >= 3 and len(x_before) >= 1 and len(x_after) >= 1 and len(y_before) >= 1 and len(y_after) >= 1:
    if best_bounce_index > 0:
        try:
            ball_positions = np.array(ball_positions)
            ballx_before = ball_positions[:best_bounce_index+1, 0] * 100  # Convert to cm
            bally_before = ball_positions[:best_bounce_index+1, 1] * 100  # Convert to cm
            ballx_after = ball_positions[best_bounce_index:, 0] * 100  # Convert to cm
            bally_after = ball_positions[best_bounce_index:, 1] * 100  # Convert to cm
            print("ballx_before: ", ballx_before, "bally_before: ", bally_before, "ballx_after:c", ballx_after, "bally_after: ", bally_after)

            # Fit quadratic functions with optimal bounce point
            model_before, poly_before, r2_before = fit_quadratic(ballx_before, bally_before)
            model_after, poly_after, r2_after = fit_quadratic(ballx_after, bally_after)

            # Generate smooth curves for plotting
            x_smooth_before = np.linspace(min(ballx_before), max(ballx_before), 100).reshape(-1, 1)
            x_smooth_after = np.linspace(min(ballx_after), max(ballx_after), 100).reshape(-1, 1)

            y_smooth_before = model_before.predict(poly_before.transform(x_smooth_before))
            y_smooth_after = model_after.predict(poly_after.transform(x_smooth_after))

            print("Y smooth:", y_smooth_after)

            # Create the plot
            plt.figure(figsize=(12, 6))
            print("plot created")
            # Plot original data points
            plt.scatter(ballx_before, bally_before, color='blue', alpha=0.5, label='Data (Before Bounce)')
            plt.scatter(ballx_after, bally_after, color='red', alpha=0.5, label='Data (After Bounce)')
            print("plot points")
            # Plot fitted curves
            plt.plot(x_smooth_before, y_smooth_before, 'b-', linewidth=2, label=f'Quadratic Fit (Before)')
            plt.plot(x_smooth_after, y_smooth_after, 'r-', linewidth=2, label=f'Quadratic Fit (After)')
            print("plot fitted")
            # Mark the bounce point
            bounce_x = ball_positions[best_bounce_index] * 100  # Convert to cm
            bounce_y = ball_positions[best_bounce_index] * 100  # Convert to cm
            plt.plot(bounce_x, bounce_y, 'go', markersize=12, label='Detected Bounce Point')
            print("plot bounce")
            # Customize the plot
            plt.title('Tennis Ball Trajectory with Automatic Bounce Detection', fontsize=14)
            plt.xlabel('X Position (cm)', fontsize=12)
            plt.ylabel('Y Position (cm)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            print("customize plot")
            # Adjust axis limits based on data
            plt.xlim(min(ball_positions[:, 0])*100 - 10, max(ball_positions[:, 0])*100 + 10)
            plt.ylim(min(min(bally_before), min(bally_after)) - 10, max(max(bally_before), max(bally_after)) + 10)
            print("adjust axis")
            # Add court baseline (ground)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            print("add baseline")
            plt.tight_layout()
            plt.show() 
            print("plotted")
        except Exception as e:
            print("error")
    else:
        print("no bounce")
else:
    print("ballpositions error")
            
cap.release()
cv2.destroyAllWindows()