import cv2
import numpy as np
import matplotlib.pyplot as plt

def foreground_equalization(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to get the mask
    _, mask = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
    
    # Get the foreground (apply the mask to the grayscale image)
    foreground = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    # Equalize the histogram of the foreground
    equalized_foreground = cv2.equalizeHist(foreground)
    
    # Get the background (apply the inverse of the mask)
    background = cv2.bitwise_and(gray_image, gray_image, mask=cv2.bitwise_not(mask))
    
    # Combine the equalized foreground with the background
    result = cv2.add(background, equalized_foreground)
    
    return result

# Load the image
image = cv2.imread('image dataset/jeniffer.jpg')

# Check if the image is loaded properly
if image is None:
    print("Error: Unable to load image.")
else:
    # Apply foreground equalization
    equalized_image = foreground_equalization(image)

    # Display the result
    cv2.imshow('Equalized Foreground', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
