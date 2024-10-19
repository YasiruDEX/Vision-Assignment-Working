import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the brain proton density image
brain_proton_img = cv2.imread('image dataset/brain_proton_density_slice.png', cv2.IMREAD_GRAYSCALE)

# Display image shape
print(f"Image Shape: {brain_proton_img.shape}")

# Define points for white and gray matter
white_matter_point = (130, 150)
gray_matter_point = (140, 95)

# Get pixel intensity values at the selected points
white_matter_intensity = brain_proton_img[white_matter_point[1], white_matter_point[0]]
gray_matter_intensity = brain_proton_img[gray_matter_point[1], gray_matter_point[0]]

# Print the pixel intensity values
print(f"White Matter Intensity: {white_matter_intensity}")
print(f"Gray Matter Intensity: {gray_matter_intensity}")

# Function to transform pixel intensities for white and gray matter
def accentuate_matter(image):
    # Create a copy of the image to apply transformations
    transformed_image = np.copy(image)
    
    # Apply transformation for gray matter (186 <= pixel <= 250)
    gray_matter_mask = (image >= 186) & (image <= 250)
    transformed_image[gray_matter_mask] = np.clip(1.75 * image[gray_matter_mask] + 30, 0, 255)
    
    # Apply transformation for white matter (150 <= pixel <= 185)
    white_matter_mask = (image >= 150) & (image <= 185)
    transformed_image[white_matter_mask] = np.clip(1.55 * image[white_matter_mask] + 22.5, 0, 255)
    
    return transformed_image, white_matter_mask, gray_matter_mask

# Apply the transformation to the brain image
transformed_brain_img, white_matter_mask, gray_matter_mask = accentuate_matter(brain_proton_img)

# Plot all the results in a single window
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axs[0, 0].imshow(brain_proton_img, cmap='gray')
axs[0, 0].scatter(white_matter_point[0], white_matter_point[1], color='red', label='White Matter')
axs[0, 0].scatter(gray_matter_point[0], gray_matter_point[1], color='blue', label='Gray Matter')
axs[0, 0].legend()
axs[0, 0].set_title("Original Brain Proton Density Image")

# Transformed image
axs[0, 1].imshow(transformed_brain_img, cmap='gray')
axs[0, 1].set_title("Transformed Brain Image")

# White matter mask
axs[0, 2].imshow(white_matter_mask, cmap='gray')
axs[0, 2].set_title("White Matter Mask")

# Gray matter mask
axs[1, 0].imshow(gray_matter_mask, cmap='gray')
axs[1, 0].set_title("Gray Matter Mask")

# Intensity transformation curves
x_vals = np.arange(0, 256)  # Intensity range (0-255)

# Apply the transformation only within specific ranges
white_matter_transformed = np.array([1.55 * x + 22.5 if 150 <= x <= 185 else x for x in x_vals])
gray_matter_transformed = np.array([1.75 * x + 30 if 186 <= x <= 250 else x for x in x_vals])

# Plot intensity transformation curves
axs[1, 1].plot(x_vals, white_matter_transformed, label='White Matter Transformation', color='blue')
axs[1, 1].plot(x_vals, gray_matter_transformed, label='Gray Matter Transformation', color='green')
axs[1, 1].plot(x_vals, x_vals, label='Original Intensity', linestyle='--', color='red')  # Identity line
axs[1, 1].set_title('Intensity Transformations')
axs[1, 1].set_xlabel('Original Intensity')
axs[1, 1].set_ylabel('Transformed Intensity')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Hide the empty subplot
axs[1, 2].axis('off')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
