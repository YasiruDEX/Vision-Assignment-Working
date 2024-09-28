import cv2
import numpy as np

# Load the image
image = cv2.imread('image dataset/einstein.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale for Sobel

# Define Sobel kernels (manual implementation)
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]])

# Apply Sobel kernels using filter2D
grad_x = cv2.filter2D(image, -1, sobel_x)
grad_y = cv2.filter2D(image, -1, sobel_y)

# Combine the gradients using the magnitude formula
sobel_combined = np.sqrt(np.square(grad_x) + np.square(grad_y))

# Normalize the result to fit in the range [0, 255] for display
sobel_combined_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8 format for display
sobel_combined_normalized = np.uint8(sobel_combined_normalized)

# Show the images
cv2.imshow('Sobel X', grad_x)
cv2.imshow('Sobel Y', grad_y)
cv2.imshow('Sobel Combined', sobel_combined_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
