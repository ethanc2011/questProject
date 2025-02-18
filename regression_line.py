import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit

# Simulation parameters
g = 9.81  # gravity in m/s^2
t_frame = 1 / 120  # time between frames (120 FPS)
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
t = 0.0
x = x0
y = y0
vx = v0x
vy = v0y
max_bounce = 2
bounce = 0

# Simulation loop
for i in range(N_frames):
    # Append current positions
    t_array.append(t)
    x_array.append(x)
    y_array.append(y)

    # Update time
    t += t_frame

    # Update positions
    x = x0 + vx * t
    y = y0 + vy * t - 0.5 * g * t ** 2
    print("x: ", x)
    print("y: ", y)

    if y <= 0:
        # Ball hits the ground
        y = 0
        vy = -vy * e  # reverse and reduce vertical velocity
        y0 = y
        x0 = x
        t = 0  # reset time after bounce
        bounce+=1
    if bounce>max_bounce:
        break

# Convert lists to numpy arrays
t_array = np.array(t_array)
x_array = np.array(x_array)
y_array = np.array(y_array)

# Add noise to simulate measurement error
noise_level = 0.02  # noise level in meters
x_noisy = x_array + np.random.normal(0, noise_level, size=x_array.shape)
y_noisy = y_array + np.random.normal(0, noise_level, size=y_array.shape)

# Quadratic function for regression
def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

# Identify indices where bounces occur
bounce_indices = np.where(y_noisy == 0)[0]

# Segment the data based on bounces
segments = []
start_idx = 0

for idx in bounce_indices:
    end_idx = idx + 1  # Include the point where y = 0
    segments.append((start_idx, end_idx))
    start_idx = idx
# Add the last segment after the final bounce
segments.append((start_idx, len(x_noisy)))

# Perform quadratic regression on each segment
regressions = []
for start, end in segments:
    xi = x_noisy[start:end]
    yi = y_noisy[start:end]

    if len(xi) >= 5:
        # Perform quadratic regression
        params, _ = curve_fit(quadratic, xi, yi)
        regressions.append((params, xi, yi))
    else:
        # Not enough data points to perform regression
        regressions.append((None, xi, yi))

# Find intersection points between consecutive regression curves
bounce_points = []
for i in range(len(regressions) - 1):
    params1, xi1, yi1 = regressions[i]
    params2, xi2, yi2 = regressions[i + 1]

    if params1 is not None and params2 is not None:
        # Solve for intersection of two quadratics
        a_diff = params1[0] - params2[0]
        b_diff = params1[1] - params2[1]
        c_diff = params1[2] - params2[2]

        discriminant = b_diff ** 2 - 4 * a_diff * c_diff

        if discriminant >= 0 and a_diff != 0:
            sqrt_discriminant = np.sqrt(discriminant)
            x_intersect1 = (-b_diff + sqrt_discriminant) / (2 * a_diff)
            x_intersect2 = (-b_diff - sqrt_discriminant) / (2 * a_diff)

            x_min = max(min(xi1), min(xi2))
            x_max = min(max(xi1), max(xi2))
            x_intersects = [x for x in [x_intersect1, x_intersect2] if x_min <= x <= x_max]

            if x_intersects:
                x_bounce = x_intersects[0]
                y_bounce = quadratic(x_bounce, *params1)
                bounce_points.append((x_bounce, y_bounce))
            else:
                bounce_points.append((None, None))
        else:
            bounce_points.append((None, None))
    else:
        bounce_points.append((None, None))

# Prepare figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Initial plot
line_data, = ax.plot([], [], 'bo', label='Data Points')
bounce_marker, = ax.plot([], [], 'gx', markersize=10, label='Bounce Points')

ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.legend()
ax.set_ylim([min(y_noisy) - 0.5, max(y_noisy) + 0.5])
ax.set_xlim([min(x_noisy) - 0.5, max(x_noisy) + 0.5])

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(x_noisy) - 1, valinit=0, valstep=1)

def update(val):
    frame = int(slider.val)
    x_plot = x_noisy[:frame + 1]
    y_plot = y_noisy[:frame + 1]
    line_data.set_data(x_plot, y_plot)

    # Clear previous regression lines
    lines_to_remove = [line for line in ax.get_lines() if line not in [line_data, bounce_marker]]
    for line in lines_to_remove:
        ax.lines.remove(line)

    # Plot regression lines up to current frame
    for idx, (params, xi_segment, yi_segment) in enumerate(regressions):
        segment_indices = np.where((x_noisy >= xi_segment[0]) & (x_noisy <= xi_segment[-1]))[0]
        if len(segment_indices) == 0 or segment_indices[0] > frame:
            continue
        segment_indices = segment_indices[segment_indices <= frame]
        xi_fit = np.linspace(xi_segment[0], xi_segment[-1], 100)
        if params is not None:
            yi_fit = quadratic(xi_fit, *params)
            ax.plot(xi_fit, yi_fit, 'r-')

    # Plot bounce points if within current frame
    bounce_x = []
    bounce_y = []
    for idx, (x_bounce, y_bounce) in enumerate(bounce_points):
        if x_bounce is not None and x_bounce <= x_noisy[frame]:
            bounce_x.append(x_bounce)
            bounce_y.append(y_bounce)
    bounce_marker.set_data(bounce_x, bounce_y)

    fig.canvas.draw_idle()

slider.on_changed(update)

# Initial update
update(0)

plt.show()