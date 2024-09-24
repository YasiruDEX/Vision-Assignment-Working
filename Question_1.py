import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('image dataset/emma.jpg', 0)

# Define control points (x, y) where x is input intensity and y is output intensity
c = np.array([(0, 0), (50, 50), (100, 100), (150, 255), (255, 255)])

# Create the transformation based on control points
# Map input intensity to output intensity using interpolation
transform = np.interp(np.arange(256), c[:, 0], c[:, 1]).astype('uint8')

# Apply the transformation using cv2.LUT (Look-Up Table)
image_transformed = cv2.LUT(image, transform)

# Plot the original and transformed images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(image_transformed, cmap='gray')
plt.title('Transformed Image')

plt.show()
