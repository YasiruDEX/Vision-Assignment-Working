import numpy as np
import matplotlib.pyplot as plt

# Function to fit a line given two points
def fit_line(p1, p2):
    delta = p2 - p1
    norm = np.linalg.norm(delta)
    if norm == 0:
        return None
    normal_vector = np.array([-delta[1], delta[0]]) / norm
    d = np.dot(normal_vector, p1)
    return normal_vector, d

# Function to compute the distance of a point from the line
def point_line_distance(point, normal_vector, d):
    return abs(np.dot(normal_vector, point) - d)

# RANSAC algorithm to fit a line
def ransac_line(X, num_iterations=1000, distance_threshold=0.3, min_inliers=50):
    best_inliers = []
    best_model = None

    for _ in range(num_iterations):
        # Randomly select two points to define a line
        sample_points = X[np.random.choice(X.shape[0], 2, replace=False)]
        p1, p2 = sample_points

        # Fit a line to the two points
        line_model = fit_line(p1, p2)
        if line_model is None:
            continue

        normal_vector, d = line_model

        # Compute inliers (points close to the line)
        inliers = []
        for point in X:
            distance = point_line_distance(point, normal_vector, d)
            if distance < distance_threshold:
                inliers.append(point)

        # If this model has more inliers, update the best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = line_model

    return best_model, np.array(best_inliers)

# Function to fit a circle given three points
def fit_circle(p1, p2, p3):
    A = np.array([p1, p2, p3])
    b = np.sum(A**2, axis=1)
    A = np.hstack((2*A, np.ones((3, 1))))
    
    # Solve for [x, y, radius^2]
    x, y, r_sq = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = np.sqrt(r_sq + x**2 + y**2)
    
    return np.array([x, y]), radius

# Function to compute radial distance of a point from the circle
def point_circle_distance(point, center, radius):
    return abs(np.linalg.norm(point - center) - radius)

# RANSAC algorithm to fit a circle
def ransac_circle(X, num_iterations=1000, distance_threshold=0.3, min_inliers=30):
    best_inliers = []
    best_model = None

    for _ in range(num_iterations):
        # Randomly select three points to define a circle
        sample_points = X[np.random.choice(X.shape[0], 3, replace=False)]
        p1, p2, p3 = sample_points

        # Fit a circle to the three points
        circle_model = fit_circle(p1, p2, p3)
        if circle_model is None:
            continue

        center, radius = circle_model

        # Compute inliers (points close to the circle)
        inliers = []
        for point in X:
            distance = point_circle_distance(point, center, radius)
            if distance < distance_threshold:
                inliers.append(point)

        # If this model has more inliers, update the best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = circle_model

    return best_model, np.array(best_inliers)

# Generate noisy point set
N = 100
half_n = N // 2
r = 10
x0_gt, y0_gt = 2, 3  # Circle center
s = r / 16
t = np.random.uniform(0, 2 * np.pi, half_n)
n = s * np.random.randn(half_n)
x, y = x0_gt + (r + n) * np.cos(t), y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

s = 1.0
m, b = -1, 2
x = np.linspace(-12, 12, half_n)
y = m * x + b + s * np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

# Combine circle and line points
X = np.vstack((X_circ, X_line))

# Apply RANSAC to estimate line
line_model, line_inliers = ransac_line(X)

# Apply RANSAC to estimate circle
X_remnant = np.array([point for point in X if point not in line_inliers])
circle_model, circle_inliers = ransac_circle(X_remnant)

# Plot the final diagram with best-fit and RANSAC estimates
fig, ax = plt.subplots(figsize=(8, 8))

# Plot original points
ax.scatter(X[:, 0], X[:, 1], label='All points', color='blue')

# Plot line inliers
ax.scatter(line_inliers[:, 0], line_inliers[:, 1], color='yellow', label='Line inliers')

# Plot circle inliers
ax.scatter(circle_inliers[:, 0], circle_inliers[:, 1], color='cyan', label='Circle inliers')

# Plot best-fit line
if line_model:
    normal_vector, d = line_model
    x_vals = np.array([-12, 12])
    y_vals = -(normal_vector[0] * x_vals - d) / normal_vector[1]
    ax.plot(x_vals, y_vals, color='magenta', label='RANSAC line')

# Plot ground truth line
y_gt_vals = m * x_vals + b
ax.plot(x_vals, y_gt_vals, color='purple', label='Ground truth line')

# Plot best-fit circle
if circle_model:
    center, radius = circle_model
    circle = plt.Circle(center, radius, color='magenta', fill=False, label='RANSAC circle')
    ax.add_patch(circle)

# Plot ground truth circle
circle_gt = plt.Circle((x0_gt, y0_gt), r, color='green', fill=False, label='Ground truth circle')
ax.add_patch(circle_gt)

# Mark ground truth circle center
ax.plot((x0_gt), (y0_gt), '+', color='green')

# Add legends and plot limits
plt.legend()
plt.xlim(-14, 14)
plt.ylim(-14, 16)
plt.title('Line and Circle Fitting using RANSAC and Ground Truth')
plt.show()
