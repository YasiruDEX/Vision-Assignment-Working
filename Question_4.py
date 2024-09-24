import cv2
import numpy as np
import matplotlib.pyplot as plt# Load brain image

# Split into HSV and apply vibrance formula
image = cv2.imread('image dataset/spider.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def vibrance_transform(saturation, a, sigma):
    return np.minimum(saturation + a * 128 * np.exp(-((saturation - 128) ** 2) / (2 * sigma ** 2)), 255)

# Apply vibrance
saturation_plane = hsv_image[:,:,1]
enhanced_saturation = vibrance_transform(saturation_plane, a=0.8, sigma=70)

# Combine and display result
hsv_image[:,:,1] = enhanced_saturation
enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
cv2.imshow('Enhanced Vibrance', enhanced_image)
cv2.waitKey(0)
