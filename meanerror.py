import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulation parameters
g = 9.81  # gravity in m/s^2
t_frame = 1 / 60  # time between frames (120 FPS)
e = 0.8  # coefficient of restitution (energy loss on bounce)
N_frames = 400  # total number of frames to simulate

# Initial conditions
x0 = 0.0
y0 = 1.0  # initial height in meters
v0x = 5.0  # initial horizontal velocity in m/s
v0y = 3.0  # initial vertical velocity in m/s

# Arrays to store positions
t_array = []
x_array = []
y_array = []

# Initialize variables
time = 0.0
x = x0
y = y0
vx = v0x
vy = v0y
max_bounce = 2
bounce = 0


# Simulation loop
for i in range(N_frames):
    # Append current positions
    t_array.append(time)
    x_array.append(x)
    y_array.append(y)

    # Update time
    time += t_frame

    # Update positions
    x = x0 + vx * time
    y = y0 + vy * time - 0.5 * g * time ** 2

    if y <= 0:
        # Ball hits the ground
        y = 0
        vy = -vy * e  # reverse and reduce vertical velocity
        y0 = y
        x0 = x
        time = 0.0  # reset local time after bounce
        bounce += 1
        if bounce > max_bounce:
            break

t_array = np.array(t_array)
x_array = np.array(x_array)
y_array = np.array(y_array)

noise_level = 0.05  # noise level in meters
x_noisy = x_array + np.random.normal(0, noise_level, size=x_array.shape)
y_noisy = y_array + np.random.normal(0, noise_level, size=y_array.shape)

# # Original data points
# time = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112,  # before bounce
#         120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224]  # after bounce

# x = [200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 550,  # before bounce
#      575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900]  # after bounce

# y = [250, 240, 228, 215, 200, 183, 165, 145, 123, 100, 75, 48, 20,  # before bounce
#      15, 28, 40, 50, 58, 65, 70, 73, 75, 74, 72, 68, 62, 55, 46]  # after bounce

# # Add noise to the data
# np.random.seed(42)  # for reproducibility
# x_noise = np.random.normal(0, 5, len(x))  # Standard deviation of 5 for x coordinates
# y_noise = np.random.normal(0, 3, len(y))  # Standard deviation of 3 for y coordinates

# # Add noise to coordinates
# x = [int(xi + noise) for xi, noise in zip(x, x_noise)]
# y = [int(yi + noise) for yi, noise in zip(y, y_noise)] 

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

results = []
min_points = 3  # minimum number of points required for fitting
len_data = len(x_array)
best_bounce_index = -1
best_mse = (float('inf'))  # We want to maximize the sum of R² scores

for bounce_idx in range(min_points, len_data - min_points):
    # Split data
    # x_before = np.array(x_noisy[:bounce_idx+1]).reshape(1, -1)
    # y_before = np.array(y_noisy[:bounce_idx+1]).reshape(1, -1)
    # x_after = np.array(x_noisy[bounce_idx:]).reshape(1, -1)
    # y_after = np.array(y_noisy[bounce_idx:]).reshape(1, -1)
    # x_before = (x_noisy[:bounce_idx+1]).reshape(1, -1)
    # y_before = (y_noisy[:bounce_idx+1]).reshape(1, -1)
    # x_after = (x_noisy[bounce_idx:]).reshape(1, -1)
    # y_after = (y_noisy[bounce_idx:]).reshape(1, -1)
    x_before = (x_noisy[:bounce_idx+1]).reshape(-1, 1)
    y_before = (y_noisy[:bounce_idx+1]).reshape(-1, 1)
    x_after = (x_noisy[bounce_idx:]).reshape(-1, 1)
    y_after = (y_noisy[bounce_idx:]).reshape(-1, 1)

    # Ensure there are enough points
    if len(x_before) < min_points or len(x_after) < min_points:
        continue

    # Fit both segments
    model_before, poly_before, _ = fit_quadratic(x_before, y_before)
    model_after, poly_after, _ = fit_quadratic(x_after, y_after)

    y_predicted_before = model_before.predict(poly_before.transform(x_before))
    y_predicted_after = model_after.predict(poly_after.transform(x_after))

    y_error_before = mean_square_error(y_before, y_predicted_before)
    y_error_after = mean_square_error(y_after, y_predicted_after)

    meansquarederror = y_error_before + y_error_after

    if meansquarederror < best_mse: 
            best_mse = meansquarederror
            best_bounce_index = bounce_idx
            print("found")

if best_bounce_index >= 0:
    print(f"best_bounce_x: {x_noisy[best_bounce_index]}, best_bounce_y: {y_noisy[best_bounce_index]}, best_mean_squared_error: {best_mse}")
else:
    print("no bounce index")

# Fit using the best bounce index
x_before = np.array(x_noisy[:best_bounce_index+1]) * 100  # Convert to cm
y_before = np.array(y_noisy[:best_bounce_index+1]) * 100  # Convert to cm
x_after = np.array(x_noisy[best_bounce_index:]) * 100  # Convert to cm
y_after = np.array(y_noisy[best_bounce_index:]) * 100  # Convert to cm

# Fit quadratic functions with optimal bounce point
model_before, poly_before, r2_before = fit_quadratic(x_before, y_before)
model_after, poly_after, r2_after = fit_quadratic(x_after, y_after)

# Get coefficients for both quadratic equations
coef_before = model_before.coef_[0]  # [a, b, c] for ax^2 + bx + c
intercept_before = model_before.intercept_
coef_after = model_after.coef_[0]    # [a, b, c] for ax^2 + bx + c
intercept_after = model_after.intercept_

# Create functions for both curves
def curve_before(x):
    return (coef_before[2] * x**2 + coef_before[1] * x + coef_before[0])

def curve_after(x):
    return (coef_after[2] * x**2 + coef_after[1] * x + coef_after[0])

# Find intersection point using numerical method
x_range = np.linspace(min(x_before), max(x_after), 1000)
y_before_curve = curve_before(x_range)
y_after_curve = curve_after(x_range)
intersect_idx = np.argmin(np.abs(y_before_curve - y_after_curve))
bounce_x = x_range[intersect_idx]
bounce_y = curve_before(bounce_x)

# Generate smooth curves for plotting
x_smooth_before = np.linspace(min(x_before), max(x_before), 100).reshape(-1, 1)
x_smooth_after = np.linspace(min(x_after), max(x_after), 100).reshape(-1, 1)

y_smooth_before = model_before.predict(poly_before.transform(x_smooth_before))
y_smooth_after = model_after.predict(poly_after.transform(x_smooth_after))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot original data points
plt.scatter(x_before, y_before, color='blue', alpha=0.5, label='Data (Before Bounce)')
plt.scatter(x_after, y_after, color='red', alpha=0.5, label='Data (After Bounce)')

# Plot fitted curves
plt.plot(x_smooth_before, y_smooth_before, 'b-', linewidth=2, label=f'Quadratic Fit (Before)')
plt.plot(x_smooth_after, y_smooth_after, 'r-', linewidth=2, label=f'Quadratic Fit (After)')

# Mark the intersection point
plt.plot(bounce_x, bounce_y, 'go', markersize=12, label='Calculated Bounce Point')

# Customize the plot
plt.title('Tennis Ball Trajectory with Automatic Bounce Detection', fontsize=14)
plt.xlabel('X Position (cm)', fontsize=12)
plt.ylabel('Y Position (cm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Adjust axis limits based on data
plt.xlim(min(x_noisy)*100 - 10, max(x_noisy)*100 + 10)
plt.ylim(min(y_noisy)*100 - 10, max(y_noisy)*100 + 10)

# Add court baseline (ground)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()