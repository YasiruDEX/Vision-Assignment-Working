import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/spider.png')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split into hue, saturation, and value planes
hue, saturation, value = cv2.split(hsv_image)

# Parameters for the intensity transformation
sigma = 70
a = 0.5  # You can adjust this to find a visually pleasing result

# Define the intensity transformation function
def vibrance_transformation(x, a, sigma):
    return np.clip(x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2)), 0, 255)

# Apply the transformation to the saturation plane
saturation_transformed = vibrance_transformation(saturation, a, sigma).astype(np.uint8)

# Recombine the hue, saturation, and value planes
hsv_transformed = cv2.merge([hue, saturation_transformed, value])

# Convert the transformed HSV image back to BGR color space
vibrance_enhanced_image = cv2.cvtColor(hsv_transformed, cv2.COLOR_HSV2BGR)

# Plot the original image, vibrance-enhanced image, and the transformation function
plt.figure(figsize=(15, 5))

# Display original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Display vibrance-enhanced image
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(vibrance_enhanced_image, cv2.COLOR_BGR2RGB))
plt.title(f'Vibrance Enhanced Image (a = {a})')

# Plot the intensity transformation function
x_values = np.arange(256)
y_values = vibrance_transformation(x_values, a, sigma)

plt.subplot(1, 3, 3)
plt.plot(x_values, y_values, label=f'Intensity Transformation (a = {a})')
plt.title('Intensity Transformation')
plt.xlabel('Input Intensity (x)')
plt.ylabel('Output Intensity (f(x))')
plt.grid(True)

plt.tight_layout()
plt.show()
