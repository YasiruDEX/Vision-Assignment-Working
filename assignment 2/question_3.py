import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images: architectural image and flag image
image1 = cv2.imread('data/image2.jpg')  # Architectural image
image2 = cv2.imread('data/image1.jpg')  # Flag image

# Function to select four points on the architectural image
points_image1 = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked points
        points_image1.append((x, y))
        print(f"Point selected: ({x}, {y})")

        # Draw a circle where the user clicked
        cv2.circle(image1_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 points on Architectural Image", image1_copy)

# Create a copy of the image to show the selected points
image1_copy = image1.copy()

# Show the architectural image and wait for the user to select 4 points
cv2.imshow("Select 4 points on Architectural Image", image1_copy)
cv2.setMouseCallback("Select 4 points on Architectural Image", select_points)
cv2.waitKey(0)  # Wait until all 4 points are selected
cv2.destroyAllWindows()

# Ensure exactly 4 points are selected
if len(points_image1) != 4:
    raise ValueError("You must select exactly 4 points!")

# Define 4 corresponding points in the flag image (corners of the flag image)
h_flag, w_flag = image2.shape[:2]
points_image2 = np.array([[0, 0], [w_flag, 0], [w_flag, h_flag], [0, h_flag]], dtype='float32')

# Convert selected points from image1 into numpy array
points_image1 = np.array(points_image1, dtype='float32')

# Compute the homography matrix
H, status = cv2.findHomography(points_image2, points_image1)

# Warp the flag image using the homography
warped_flag = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]))

# Create a mask of the flag
flag_mask = np.zeros_like(image1, dtype=np.uint8)
cv2.fillConvexPoly(flag_mask, points_image1.astype(int), (255, 255, 255))

# Blend the warped flag into the architectural image
blended_image = cv2.addWeighted(image1, 1, warped_flag, 0.6, 0)

# Show the final result
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
plt.title('Architectural Image with Superimposed Flag')
plt.axis('off')
plt.show()

# Optionally, save the result
cv2.imwrite('blended_image.png', blended_image)
