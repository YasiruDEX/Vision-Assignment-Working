import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply histogram equalization
def histogram_equalization(f):
    L = 256  # Number of gray levels
    M, N = f.shape  # Get image dimensions

    # Get histogram
    hist = cv2.calcHist([f], [0], None, [L], [0, L]).flatten()

    # Calculate CDF
    cdf = hist.cumsum()
    
    # Normalize CDF to the range [0, L-1]
    cdf_normalized = (L - 1) * cdf / cdf[-1]
    
    # Define transformation function
    t = np.array([cdf_normalized[k] for k in range(L)]).astype("uint8")
    
    # Apply the transformation
    return t[f]

# Load the image in grayscale
image = cv2.imread('image dataset/shells.tif', 0)  # The 0 flag ensures the image is loaded in grayscale

# Check if the image is loaded properly
if image is None:
    print("Error: Unable to load image.")
else:
    # Apply histogram equalization
    hist_equalized = histogram_equalization(image)

    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Display the original image
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Display the equalized image
    axs[0, 1].imshow(hist_equalized, cmap='gray')
    axs[0, 1].set_title('Equalized Image')
    axs[0, 1].axis('off')

    # Plot the histogram for the original image
    axs[1, 0].hist(image.ravel(), bins=256, color='blue', alpha=0.5)
    axs[1, 0].set_title('Original Image Histogram')

    # Plot the histogram for the equalized image
    axs[1, 1].hist(hist_equalized.ravel(), bins=256, color='red', alpha=0.5)
    axs[1, 1].set_title('Equalized Image Histogram')

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Show the figure
    plt.show()
