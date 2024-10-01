import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_filter(image, filter):
    rows, columns = image.shape  # Get rows and columns of the image
    filtered_image = np.zeros((rows, columns))  # Create empty image
    # Process 2D convolution
    for i in range(1, rows - 1):  # Start from 1 to avoid border issues
        for j in range(1, columns - 1):
            value = np.sum(np.multiply(filter, image[i - 1:i + 2, j - 1:j + 2]))
            filtered_image[i, j] = value
    return filtered_image

# Load the image
image = cv2.imread('image dataset/einstein.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale for Sobel

# Define Sobel filters
sobel_filter_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

sobel_filter_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

# Apply custom Sobel filters
sobel_x_custom = apply_filter(image, sobel_filter_x)
sobel_y_custom = apply_filter(image, sobel_filter_y)

# Calculate the magnitude of the gradients
sobel_combined_custom = np.sqrt(sobel_x_custom**2 + sobel_y_custom**2)

# Display the results
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Sobel Filtered Image (X Direction)')
plt.imshow(sobel_x_custom, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Sobel Filtered Image (Y Direction)')
plt.imshow(sobel_y_custom, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Sobel Magnitude Image')
plt.imshow(sobel_combined_custom, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
