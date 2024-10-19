import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the flower image
image = cv2.imread('image dataset/daisy.jpg')
if image is None:
    print("Error: Unable to load image.")
    exit()

# Create a mask for GrabCut (same size as the image, single channel, initially set to GC_BGD or GC_PR_BGD)
mask = np.zeros(image.shape[:2], np.uint8)

# Models needed for GrabCut (these are internal arrays used by the algorithm)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Create a rectangle around the flower (this is a rough estimate of the region)
rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)  # Adjust these values as needed

# Apply GrabCut segmentation
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to segment the foreground and background
# Any pixel classified as GC_FGD (1) or GC_PR_FGD (3) is treated as foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Extract the foreground using the modified mask
foreground = image * mask2[:, :, np.newaxis]

# Extract the background by inverting the mask and then applying a blur
background = image * (1 - mask2[:, :, np.newaxis])
blurred_background = cv2.GaussianBlur(background, (25, 25), 0)

# Combine the blurred background and the sharp foreground
enhanced_image = blurred_background + foreground

# Display the images: original, segmentation mask, foreground, background, and enhanced image
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(mask2, cmap='gray')
plt.title('Segmentation Mask')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.title('Foreground (Flower)')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.title('Background')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(blurred_background, cv2.COLOR_BGR2RGB))
plt.title('Blurred Background')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title('Enhanced Image (Blurred Background)')

plt.tight_layout()
plt.show()
