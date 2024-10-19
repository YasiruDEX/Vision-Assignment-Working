import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/spider.png')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split into hue, saturation, and value planes
hue, saturation, value = cv2.split(hsv_image)


# Plotting the images
plt.figure(figsize=(12, 4))

# Display the Hue channel
plt.subplot(1, 3, 1)
plt.imshow(hue, cmap='gray')
plt.title('Hue')
plt.axis('off')

# Display the Saturation channel
plt.subplot(1, 3, 2)
plt.imshow(saturation, cmap='gray')
plt.title('Saturation')
plt.axis('off')

# Display the Value channel
plt.subplot(1, 3, 3)
plt.imshow(value, cmap='gray')
plt.title('Value')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()


# Parameters for the intensity transformation
sigma = 70
a_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different values of 'a'

# Define the intensity transformation function
def vibrance_transformation(x, a, sigma):
    return np.clip(x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2)), 0, 255)

# Plot setup
plt.figure(figsize=(15, 10))

for i, a in enumerate(a_values):
    # Apply the transformation to the saturation plane
    saturation_transformed = vibrance_transformation(saturation, a, sigma).astype(np.uint8)

    # Recombine the hue, saturation, and value planes
    hsv_transformed = cv2.merge([hue, saturation_transformed, value])

    # Convert the transformed HSV image back to BGR color space
    vibrance_enhanced_image = cv2.cvtColor(hsv_transformed, cv2.COLOR_HSV2BGR)

    # Display vibrance-enhanced image
    plt.subplot(3, 3, i + 1)
    plt.imshow(cv2.cvtColor(vibrance_enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Vibrance Enhanced (a = {a})')

# Plot the intensity transformation functions for each 'a' value
x_values = np.arange(256)
plt.subplot(3, 3, 6)
for a in a_values:
    y_values = vibrance_transformation(x_values, a, sigma)
    plt.plot(x_values, y_values, label=f'a = {a}')

plt.title('Intensity Transformation')
plt.xlabel('Input Intensity (x)')
plt.ylabel('Output Intensity (f(x))')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
