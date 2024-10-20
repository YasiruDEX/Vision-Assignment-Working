import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the sunflower field image
sunflower = cv.imread('data/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)
sunflower_grey = cv.cvtColor(sunflower, cv.COLOR_BGR2GRAY)

# Set up parameters for scale-space extrema detection
min_sigma = 1.0  # Minimum sigma value (smaller values for smaller blobs)
max_sigma = 10  # Maximum sigma value (larger values for larger blobs)
num_sigma = 2   # Number of sigma values to test
threshold = 0.3  # Threshold for blob detection

circles = []
# Loop through different sigma values to detect blobs at different scales
for sigma in np.linspace(min_sigma, max_sigma, num_sigma):

    # Print the current sigma value to the console
    print(f"Current sigma value: {sigma}")

    # Apply GaussianBlur (to simulate scale-space) to the grayscale image with the current sigma
    blurred = cv.GaussianBlur(sunflower_grey, (0, 0), sigma)

    # Apply the Laplacian (LoG approximation) to find blobs
    laplacian = cv.Laplacian(blurred, cv.CV_64F)

    # Calculate the absolute Laplacian values
    abs_laplacian = np.abs(laplacian)

    # Create a binary image where blobs are detected using the threshold
    blob_mask = abs_laplacian > threshold * abs_laplacian.max()

    # Find contours in the blob mask
    contours, _ = cv.findContours(blob_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Loop through the detected contours and fit circles to them
    for contour in contours:
        if len(contour) >= 5:  # Fit enclosing circle only if enough points are in the contour
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius, sigma))

# Sort the detected circles by radius in descending order
circles.sort(key=lambda x: -x[1])

# Report the parameters of the largest circle
if circles:
    largest_circle = circles[0]
    center, radius, sigma = largest_circle

    print("Parameters of the largest circle:")
    print(f"Center: {center}")
    print(f"Radius: {radius}")
    print(f"Sigma value: {sigma}")

# Set the desired line thickness for drawn circles
line_thickness = 1

# Draw all detected circles with the specified line thickness
output_image = cv.cvtColor(sunflower_grey, cv.COLOR_GRAY2BGR)
for circle in circles:
    center, radius, _ = circle
    cv.circle(output_image, center, radius, (0, 0, 255), line_thickness)  # Red color

# Display the grayscale image with detected circles using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(output_image, cv.COLOR_BGR2RGB), cmap='gray')
plt.axis('off')
plt.title('Detected Circles')
plt.show()