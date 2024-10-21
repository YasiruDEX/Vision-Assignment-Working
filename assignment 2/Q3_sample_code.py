import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# Define constants
N = 100
half_n = N // 2
r = 10  # Radius of the circle
x0_gt, y0_gt = 2, 3  # Ground truth center of the circle
s = r / 16  # Standard deviation for noise

# Generate random points for the circle
t = np.random.uniform(0, 2 * np.pi, half_n)
n = s * np.random.randn(half_n)
x_circle = x0_gt + (r + n) * np.cos(t)
y_circle = y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x_circle.reshape(half_n, 1), y_circle.reshape(half_n, 1)))

# Generate random points for the line
s = 1.0
m, b = -1, 2  # Slope and intercept of the line
x_line = np.linspace(-12, 12, half_n)
y_line = m * x_line + b + s * np.random.randn(half_n)
X_line = np.hstack((x_line.reshape(half_n, 1), y_line.reshape(half_n, 1)))

# Combine circle and line points
X = np.vstack((X_circ, X_line))

# Plot the circle and line points
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_line[:, 0], X_line[:, 1], label='Line', color='blue')
ax.scatter(X_circ[:, 0], X_circ[:, 1], label='Circle', color='orange')

# Ground truth circle
circle_gt = plt.Circle((x0_gt, y0_gt), r, color='g', fill=False, label='Ground truth circle')
ax.add_patch(circle_gt)

# Plot ground truth center of the circle
ax.plot((x0_gt), (y0_gt), '+', color='g')

# Plot ground truth line
x_min, x_max = ax.get_xlim()
x_ = np.array([x_min, x_max])
y_ = m * x_ + b
plt.plot(x_, y_, color='m', label='Ground truth line')

# Add legend and show plot
plt.legend()
plt.show()
