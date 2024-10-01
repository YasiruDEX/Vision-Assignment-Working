import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/einstein.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale for Sobel

# Define a Sobel filter kernel (for detecting edges in x direction)
sobel_kernel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

# Apply the Sobel filter using filter2D function
sobel_x = cv2.filter2D(image, -1, sobel_kernel_x)

# Display the original image and the Sobel filtered image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel Filtered Image (X direction)')
plt.axis('off')

plt.show()
