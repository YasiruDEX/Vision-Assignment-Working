import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/highlights_and_shadows.jpg')

# Convert the image from BGR to L*a*b* color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Extract the L* channel (lightness)
L_channel, a_channel, b_channel = cv2.split(lab_image)

# Apply gamma correction to the L channel
gamma = 0.5  # Example gamma value (you can change it)
L_corrected = np.array(255 * (L_channel / 255) ** gamma, dtype='uint8')

# Merge the corrected L channel back with the original a* and b* channels
lab_corrected = cv2.merge([L_corrected, a_channel, b_channel])

# Convert the corrected L*a*b* image back to BGR for display
image_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)

# Plot the original and gamma corrected images
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image_corrected, cv2.COLOR_BGR2RGB))
plt.title(f'Gamma Corrected Image (γ={gamma})')

# Plot histograms for original and gamma-corrected L* channel
plt.subplot(2, 2, 3)
plt.hist(L_channel.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Original L* Channel Histogram')

plt.subplot(2, 2, 4)
plt.hist(L_corrected.ravel(), bins=256, range=(0, 256), color='black')
plt.title(f'Gamma Corrected L* Channel Histogram (γ={gamma})')

plt.tight_layout()
plt.show()
