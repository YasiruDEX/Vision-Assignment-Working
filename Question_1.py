import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('image dataset/emma.jpg', 0)

# Define control points
c = np.array([(50, 50), (50, 100), (150, 255), (150, 150)])

# Create the transformation based on control points
t1 = np.linspace(0, c[0, 1], c[0, 0] + 1).astype('uint8')   # from (0, 0) to (50, 50)
t2 = np.linspace(c[0, 1], c[1, 1], c[1, 0] - c[0, 0]).astype('uint8')  # from (50, 50) to (50, 100)
t3 = np.linspace(c[1, 1], c[2, 1], c[2, 0] - c[1, 0]).astype('uint8')  # from (50, 100) to (150, 255)
t4 = np.linspace(c[2, 1], c[3, 1], c[3, 0] - c[2, 0]).astype('uint8')  # from (150, 255) to (150, 150)
t5 = np.linspace(c[3, 1], 255, 255 - c[3, 0]).astype('uint8')  # from (150, 150) to (255, 255)

# Concatenate to form the full transformation function
transform = np.concatenate((t1, t2, t3, t4, t5))

# Manually define piecewise linear transformation based on control points
for i in range(1, len(c)):
    x_start, y_start = c[i - 1]
    x_end, y_end = c[i]
    
    # Define the slope between the two points
    slope = (y_end - y_start) / (x_end - x_start)
    
    # Apply the transformation to the corresponding range
    for x in range(x_start, x_end):
        transform[x] = np.clip(y_start + int(slope * (x - x_start)), 0, 255)

# Ensure the last point (255, 255) is mapped correctly
transform[255] = 255

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
