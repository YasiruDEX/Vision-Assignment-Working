import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (assuming it's in BGR format)
jeniffer_img = cv2.imread('image dataset/jeniffer.jpg')

# Convert the image to HSV color space
jeniffer_hsv_img = cv2.cvtColor(jeniffer_img, cv2.COLOR_BGR2HSV)

# Split the image into HSV planes
jeniffer_hue_plane = jeniffer_hsv_img[:, :, 0]
jeniffer_saturation_plane = jeniffer_hsv_img[:, :, 1]
jeniffer_value_plane = jeniffer_hsv_img[:, :, 2]

# Create the foreground mask using a threshold on the saturation plane
foreground_mask = jeniffer_saturation_plane > 15

# Extract the foreground only using cv2.bitwise_and
foreground_only = cv2.bitwise_and(jeniffer_img, jeniffer_img, mask=foreground_mask.astype(np.uint8))

# Extract the background from the original image
jeniffer_background_bgr = jeniffer_img - foreground_only

# Get the foreground image in the Value plane
jeniffer_foreground_value = cv2.bitwise_and(jeniffer_value_plane, jeniffer_value_plane, mask=foreground_mask.astype(np.uint8))

# Compute the histogram of the value plane
hist_v = cv2.calcHist([jeniffer_foreground_value], [0], foreground_mask.astype(np.uint8), [256], [0, 256])

# Obtain cumulative sum of histograms
cum_hist_v = np.cumsum(hist_v)

# Calculate pixel count in the mask
pixel_count = foreground_mask.astype(np.uint8).sum()

# Look up table for value channel equalization
def equalize_histogram(cum_hist, pixel_count):
    return ((cum_hist / pixel_count) * 255).astype(np.uint8)

lut_v = equalize_histogram(cum_hist_v, pixel_count)

# Apply equalization to the foreground value plane
jeniffer_equalized_foreground_value = cv2.LUT(jeniffer_foreground_value, lut_v)

# Merge the equalized value plane with the original hue and saturation planes
jeniffer_equalized_foreground_hsv_img = cv2.merge((jeniffer_hue_plane, jeniffer_saturation_plane, jeniffer_equalized_foreground_value))

# Convert the image back to BGR color space
jeniffer_equalised_foreground_bgr_img = cv2.cvtColor(jeniffer_equalized_foreground_hsv_img, cv2.COLOR_HSV2BGR)

# Combine equalized foreground with the original background
jeniffer_result = cv2.add(jeniffer_equalised_foreground_bgr_img, jeniffer_background_bgr)

# Compute histogram for the original and equalized value planes
hist_v_orig = cv2.calcHist([jeniffer_foreground_value], [0], None, [256], [0, 256])
hist_v_eq = cv2.calcHist([jeniffer_equalized_foreground_value], [0], None, [256], [0, 256])

# Display results
plt.figure(figsize=(20, 10))

# Original Image
plt.subplot(231)
plt.imshow(cv2.cvtColor(jeniffer_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Foreground Mask
plt.subplot(232)
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask')

# Foreground Only
plt.subplot(233)
plt.imshow(cv2.cvtColor(foreground_only, cv2.COLOR_BGR2RGB))
plt.title('Foreground Only')

# Equalized Foreground
plt.subplot(234)
plt.imshow(cv2.cvtColor(jeniffer_equalised_foreground_bgr_img, cv2.COLOR_BGR2RGB))
plt.title('Equalized Foreground')

# Final Result
plt.subplot(235)
plt.imshow(cv2.cvtColor(jeniffer_result, cv2.COLOR_BGR2RGB))
plt.title('Final Result')

# Histograms
# plt.subplot(236)
# # plt.plot(hist_v_orig, color='blue', label='Original Histogram')
# # plt.plot(hist_v_eq, color='red', label='Equalized Histogram')
# plt.title('Value Plane Histograms')
# plt.xlabel('Intensity Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid()

plt.tight_layout()
plt.show()
