import cv2
import numpy as np
import matplotlib.pyplot as plt# Load brain image

brain_image = cv2.imread('image dataset/brain_proton_density_slice.png', 0)

# Example for accentuating white matter
def accentuate_white_matter(image):
    return np.clip(image * 1.2 + 30, 0, 255).astype(np.uint8)

# Apply to accentuate gray matter similarly by adjusting the transformation
white_matter = accentuate_white_matter(brain_image)

# Plot the transformation
plt.imshow(white_matter, cmap='gray')
plt.title('White Matter Accentuated')
plt.show()
