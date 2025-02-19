import cv2
import numpy as np

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Reset when 'r' is pressed
        if cv2.waitKey(1) & 0xFF == ord('r'):
            image_copy = image.copy()
            points = []
    
    cv2.destroyAllWindows()
    return points if len(points) == 2 else None

if __name__ == "__main__":
    # Example usage with a video frame
    cap = cv2.VideoCapture('/Users/ethancai/Downloads/grigor.mp4')  # or video file path
    ret, frame = cap.read()
    if ret:
        coordinates = draw_line(frame)
        if coordinates:
            print("Final line coordinates:", coordinates)
    cap.release()
