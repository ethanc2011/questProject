import numpy as np

# Coefficients of the two quadratics
a1, b1, c1 = -1, -4, 2  # Example coefficients for the first quadratic
a2, b2, c2 = 0.5, -2.5, 1.5  # Example coefficients for the second quadratic

# Coefficients of the resulting quadratic equation
A = a1 - a2
B = b1 - b2
C = c1 - c2

if A == 0:
    if B == 0:
        if C == 0:
            print("The curves are identical (infinite intersections).")
        else:
            print("The curves are parallel (no intersections).")
    else:
        x = -C / B
        y = a1 * x**2 + b1 * x + c1
        print("Intersection point:", (x, y))
else:
    # Proceed with discriminant check and quadratic solution
    discriminant = B**2 - 4*A*C

    if discriminant < 0:
        print("No real intersection points (discriminant < 0).")
    else:
        roots = np.roots([A, B, C])
        intersection_points = [(x, a1*x**2 + b1*x + c1) for x in roots if np.isreal(x)]
        print("Intersection points:", intersection_points)