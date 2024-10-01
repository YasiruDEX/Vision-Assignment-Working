def apply_filter(image, filter):
    rows, columns = image.shape  # Get rows and columns of the image
    filtered_image = np.zeros((rows, columns))  # Create empty image
    # Process 2D convolution
    for i in range(1, rows - 1):  # Start from 1 to avoid border issues
        for j in range(1, columns - 1):
            value = np.sum(np.multiply(filter, image[i - 1:i + 2, j - 1:j + 2]))
            filtered_image[i, j] = value
    return filtered_image