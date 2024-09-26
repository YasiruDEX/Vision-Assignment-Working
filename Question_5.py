import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply histogram equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Load the image in grayscale
image = cv2.imread('image dataset/shells.tif', 0)  # The 0 flag ensures the image is loaded in grayscale

# Check if the image is loaded properly
if image is None:
    print("Error: Unable to load image.")
else:
    # Apply histogram equalization
    hist_equalized = histogram_equalization(image)

    # Plot the histograms before and after equalization
    plt.figure(figsize=(10, 5))

    # Original image histogram
    plt.subplot(1, 2, 1)
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.5)
    plt.title('Original Image Histogram')

    # Equalized image histogram
    plt.subplot(1, 2, 2)
    plt.hist(hist_equalized.ravel(), bins=256, color='red', alpha=0.5)
    plt.title('Equalized Image Histogram')

    # Show both histograms
    plt.show()

    # Show the original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(hist_equalized, cmap='gray')
    plt.title('Equalized Image')

    plt.show()
