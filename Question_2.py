import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the brain proton density image (grayscale image)
image = cv2.imread('image dataset/brain_proton_density_slice.png', 0)

# Intensity transformation for white matter (enhance higher intensities)
# Define control points for white matter
white_matter_points = np.array([(0, 0), (100, 100), (200, 255), (255, 255)])

# Create transformation for white matter
white_matter_transform = np.zeros(256, dtype='uint8')

# Piecewise transformation for white matter
for i in range(1, len(white_matter_points)):
    x_start, y_start = white_matter_points[i - 1]
    x_end, y_end = white_matter_points[i]
    slope = (y_end - y_start) / (x_end - x_start)
    for x in range(x_start, x_end):
        white_matter_transform[x] = np.clip(y_start + int(slope * (x - x_start)), 0, 255)

# Intensity transformation for gray matter (enhance mid-range intensities)
# Define control points for gray matter
gray_matter_points = np.array([(0, 0), (50, 50), (150, 255), (255, 255)])

# Create transformation for gray matter
gray_matter_transform = np.zeros(256, dtype='uint8')

# Piecewise transformation for gray matter
for i in range(1, len(gray_matter_points)):
    x_start, y_start = gray_matter_points[i - 1]
    x_end, y_end = gray_matter_points[i]
    slope = (y_end - y_start) / (x_end - x_start)
    for x in range(x_start, x_end):
        gray_matter_transform[x] = np.clip(y_start + int(slope * (x - x_start)), 0, 255)

# Apply the transformations using cv2.LUT (Look-Up Table)
image_white_accentuated = cv2.LUT(image, white_matter_transform)
image_gray_accentuated = cv2.LUT(image, gray_matter_transform)

# Plot the original and transformed images for white and gray matter accentuation
plt.figure(figsize=(10, 6))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# White matter accentuated image
plt.subplot(2, 3, 2)
plt.imshow(image_white_accentuated, cmap='gray')
plt.title('White Matter Accentuated')

# Gray matter accentuated image
plt.subplot(2, 3, 3)
plt.imshow(image_gray_accentuated, cmap='gray')
plt.title('Gray Matter Accentuated')

# Plot the transformation curves for white and gray matter
plt.subplot(2, 3, 5)
plt.plot(white_matter_transform, label='White Matter Transformation')
plt.title('White Matter Intensity Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(gray_matter_transform, label='Gray Matter Transformation')
plt.title('Gray Matter Intensity Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
