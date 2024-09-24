import cv2
import numpy as np
import matplotlib.pyplot as plt# Load brain image

# Convert to L*a*b* color space
image = cv2.imread('image dataset/highlights_and_shadows.jpg')
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Apply gamma correction to L channel
def gamma_correction(L_channel, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(L_channel, table)

L_channel = lab_image[:,:,0]
gamma_corrected_L = gamma_correction(L_channel, gamma=2.0)

# Show histograms
plt.hist(L_channel.ravel(), bins=256, color='blue', alpha=0.5, label='Original L')
plt.hist(gamma_corrected_L.ravel(), bins=256, color='red', alpha=0.5, label='Gamma Corrected L')
plt.legend()
plt.show()
