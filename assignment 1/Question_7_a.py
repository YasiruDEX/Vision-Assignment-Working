
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/einstein.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale for Sobel

# Apply Sobel filter using filter2D
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction

# Combine the two gradients
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Sobel Filtered Image')
plt.imshow(sobel_combined, cmap='gray')
plt.show()
