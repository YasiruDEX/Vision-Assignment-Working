import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/einstein.png', 0)  # Load in grayscale for Sobel

# Sobel using OpenCV (for comparison)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)

# Custom Sobel filter implementation
def custom_sobel(image):
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    # Apply filters
    grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
    
    # Compute magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    return magnitude

# Apply custom Sobel filter
filtered_image = custom_sobel(image)

# Display the result
cv2.imshow('Custom Sobel', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally plot the result using matplotlib
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Custom Sobel Filter')

plt.show()
