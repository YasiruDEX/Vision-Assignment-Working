import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image dataset/jeniffer.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the image into Hue, Saturation, and Value planes
hue_plane, jeniffer_saturation_plane, value_plane = cv2.split(hsv_image)

# Display the hue, saturation, and value planes
plt.figure(figsize=(20, 10))
plt.subplot(231)
plt.imshow(hue_plane, cmap='gray')
plt.title('Hue Plane')
plt.subplot(232)
plt.imshow(jeniffer_saturation_plane, cmap='gray')
plt.title('Saturation Plane')
plt.subplot(233)
plt.imshow(value_plane, cmap='gray')
plt.title('Value Plane')

# (b) Create the foreground mask using a threshold on the saturation plane
foreground_mask = jeniffer_saturation_plane > 12
plt.subplot(234)
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask')

# (c) Get the foreground only using cv2.bitwise_and
foreground_only = cv2.bitwise_and(image, image, mask=foreground_mask.astype(np.uint8))

# Display the foreground only image
plt.subplot(235)
plt.imshow(cv2.cvtColor(foreground_only, cv2.COLOR_BGR2RGB))
plt.title("Foreground Only")

# Compute histograms for each channel (B, G, R)
hist_b = cv2.calcHist([foreground_only], [0], foreground_mask.astype(np.uint8), [256], [0, 256])
hist_g = cv2.calcHist([foreground_only], [1], foreground_mask.astype(np.uint8), [256], [0, 256])
hist_r = cv2.calcHist([foreground_only], [2], foreground_mask.astype(np.uint8), [256], [0, 256])

# (d) Obtain cumulative sum of histograms
cum_hist_b = np.cumsum(hist_b)
cum_hist_g = np.cumsum(hist_g)
cum_hist_r = np.cumsum(hist_r)

# (e) Histogram equalization for the foreground
def equalize_histogram(cum_hist, pixel_count):
    return ((cum_hist / pixel_count) * 255).astype(np.uint8)

pixel_count = foreground_mask.astype(np.uint8).sum()

# Lookup tables for each color channel
lut_b = equalize_histogram(cum_hist_b, pixel_count)
lut_g = equalize_histogram(cum_hist_g, pixel_count)
lut_r = equalize_histogram(cum_hist_r, pixel_count)

# Apply equalization to foreground
equalized_foreground = np.zeros_like(foreground_only)
equalized_foreground[:,:,0] = cv2.LUT(foreground_only[:,:,0], lut_b)
equalized_foreground[:,:,1] = cv2.LUT(foreground_only[:,:,1], lut_g)
equalized_foreground[:,:,2] = cv2.LUT(foreground_only[:,:,2], lut_r)

# (f) Extract the background and combine it with the histogram-equalized foreground
background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(foreground_mask.astype(np.uint8)))
result = cv2.add(equalized_foreground, background)

# Display the results: Original, Mask, Foreground, Equalized Foreground, and Final Result
plt.subplot(236)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Final Result (Equalized Foreground)')

plt.tight_layout()
plt.show()
