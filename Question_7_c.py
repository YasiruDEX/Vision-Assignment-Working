import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_1d_filter(image, filter):
    """ Apply a 1D filter to an image. """
    rows, columns = image.shape
    filtered_image = np.zeros((rows, columns))

    # Apply filter horizontally
    for i in range(rows):
        for j in range(1, columns - 1):  # Start from 1 to avoid border issues
            value = np.sum(np.multiply(filter, image[i, j - 1:j + 2]))
            filtered_image[i, j] = value

    # Apply filter vertically
    final_image = np.zeros((rows, columns))
    for i in range(1, rows - 1):
        for j in range(columns):
            value = np.sum(np.multiply(filter, filtered_image[i - 1:i + 2, j]))
            final_image[i, j] = value

    return final_image

# Load the image
image = cv2.imread('image dataset/einstein.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale for Sobel

# Define Sobel filters for X and Y directions (as separable)
sobel_filter_x_1d = np.array([1, 2, 1])  # Vertical component
sobel_filter_y_1d = np.array([1, 0, -1])  # Horizontal component

# Apply Sobel filters using separable convolution
sobel_x_separable = apply_1d_filter(image, sobel_filter_y_1d)  # Apply horizontal filter first
sobel_y_separable = apply_1d_filter(image.T, sobel_filter_x_1d).T  # Apply vertical filter after transposing

# Calculate the magnitude of the gradients
sobel_magnitude_separable = np.sqrt(sobel_x_separable**2 + sobel_y_separable**2)

# Define the constant multiplier
constant_multiplier = 1 / 4  # Adjust based on your specific property requirement

# Scale the results
sobel_x_scaled_separable = constant_multiplier * sobel_x_separable
sobel_y_scaled_separable = constant_multiplier * sobel_y_separable
sobel_magnitude_scaled_separable = constant_multiplier * sobel_magnitude_separable

# Display the results
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Sobel Filter X (Separable)')
plt.imshow(sobel_x_scaled_separable, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Sobel Filter Y (Separable)')
plt.imshow(sobel_y_scaled_separable, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Sobel Magnitude (Separable)')
plt.imshow(sobel_magnitude_scaled_separable, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
