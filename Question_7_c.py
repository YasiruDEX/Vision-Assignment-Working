import cv2
import numpy as np

# Load the image
image = cv2.imread('image dataset/einstein.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale for Sobel

# Define the factorized kernels
smooth_kernel = np.array([[1], [2], [1]])
diff_kernel = np.array([[1, 0, -1]])

# Apply the kernels in sequence
smoothed_image = cv2.filter2D(image, -1, smooth_kernel)  # Vertical smoothing
sobel_x = cv2.filter2D(smoothed_image, -1, diff_kernel)  # Horizontal differentiation

# Show the result
cv2.imshow('Sobel X (Factorized)', sobel_x)
cv2.waitKey(0)
cv2.destroyAllWindows()
