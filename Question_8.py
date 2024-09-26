import cv2
import numpy as np

# Function to zoom images by a factor using nearest-neighbor or bilinear interpolation
def zoom_image(image, scale, interpolation_method='nearest'):
    h, w = image.shape[:2]
    if interpolation_method == 'nearest':
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    elif interpolation_method == 'bilinear':
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("Interpolation method must be 'nearest' or 'bilinear'")

# Function to compute normalized sum of squared difference (SSD)
def compute_normalized_ssd(original, zoomed):
    # Resize the zoomed image to match the original's size for comparison
    zoomed_resized = cv2.resize(zoomed, original.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    
    # Compute SSD
    ssd = np.sum((original.astype('float') - zoomed_resized.astype('float')) ** 2)
    
    # Normalize by dividing by the number of pixels
    normalized_ssd = ssd / np.prod(original.shape[:2])
    
    return normalized_ssd

# Load original and zoomed-out versions of the images
original_image = cv2.imread('image dataset/a1q5images/im03.png', 0)  # Load the original image in grayscale
zoomed_out_image = cv2.imread('image dataset/a1q5images/im03small.png', 0)  # Load the zoomed-out version

if original_image is None or zoomed_out_image is None:
    print("Error loading images. Please check the file paths.")
else:
    # Apply zoom with factor 4 (using nearest-neighbor interpolation)
    zoom_factor = 4
    zoomed_image_nn = zoom_image(zoomed_out_image, zoom_factor, interpolation_method='nearest')
    zoomed_image_bilinear = zoom_image(zoomed_out_image, zoom_factor, interpolation_method='bilinear')

    # Compute normalized SSD between original and zoomed images
    ssd_nn = compute_normalized_ssd(original_image, zoomed_image_nn)
    ssd_bilinear = compute_normalized_ssd(original_image, zoomed_image_bilinear)

    # Display the results
    print(f"Normalized SSD (Nearest-Neighbor): {ssd_nn}")
    print(f"Normalized SSD (Bilinear Interpolation): {ssd_bilinear}")

    # Show images
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Zoomed Image (Nearest-Neighbor)', zoomed_image_nn)
    cv2.imshow('Zoomed Image (Bilinear)', zoomed_image_bilinear)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
