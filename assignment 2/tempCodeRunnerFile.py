import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import svd

def compute_homography_matrix(src_pts, dst_pts):
    """
    Compute homography matrix H from corresponding points using SVD.
    """
    if len(src_pts) < 4:
        return None

    # Construct A matrix
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0]
        u, v = dst_pts[i][0]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

    A = np.array(A)

    # Solve using SVD
    _, _, Vt = svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize the homography matrix
    H /= H[2, 2]
    
    return H

def homography_ransac(src_pts, dst_pts, thresh=4.0, max_iters=2000):
    """
    Compute homography matrix using RANSAC algorithm to handle outliers.
    """
    best_H = None
    max_inliers = 0
    best_inliers_mask = None
    num_pts = len(src_pts)

    for _ in range(max_iters):
        # Randomly select 4 points
        idx = np.random.choice(num_pts, 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        # Compute homography for these points
        H = compute_homography_matrix(src_sample, dst_sample)
        if H is None:
            continue

        # Transform all points using this homography
        src_homog = np.hstack((src_pts.reshape(-1, 2), np.ones((num_pts, 1))))
        transformed_pts = (H @ src_homog.T).T
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]

        # Calculate distances between transformed and actual destination points
        distances = np.linalg.norm(dst_pts.reshape(-1, 2) - transformed_pts, axis=1)

        # Inliers mask based on the threshold
        inliers_mask = distances < thresh
        num_inliers = np.sum(inliers_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers_mask = inliers_mask
            best_H = H

    return best_H, best_inliers_mask

def sift_and_stitch(img1_path, img5_path, gt_homography=None):
    """
    Complete pipeline:
    (a) Detect SIFT features and match between two images.
    (b) Compute homography using RANSAC and compare with ground truth.
    (c) Stitch img1.ppm onto img5.ppm.
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img5 = cv2.imread(img5_path)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

    # SIFT feature extraction
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray5, None)

    # FLANN-based matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Extract points from matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # (b) Compute homography using RANSAC
    H, inliers_mask = homography_ransac(src_pts, dst_pts)
    
    # Visualize matches
    match_img = cv2.drawMatches(img1, kp1, img5, kp2, good_matches, None, matchColor=(0, 255, 0))
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title("Matched Features (Green = Matches)")
    plt.axis('off')
    plt.show()

    # Show ground truth homography if provided
    if gt_homography is not None:
        print("\nGround Truth Homography:\n", gt_homography)
    
    print("\nComputed Homography:\n", H)

    # (c) Stitch the two images using the computed homography
    h1, w1 = img1.shape[:2]
    h5, w5 = img5.shape[:2]
    
    # Transform corners of img1
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_transformed = cv2.perspectiveTransform(corners_img1, H)

    # Compute bounds of the stitched image
    min_x = int(min(np.min(corners_transformed[:, :, 0]), 0))
    max_x = int(max(np.max(corners_transformed[:, :, 0]), w5))
    min_y = int(min(np.min(corners_transformed[:, :, 1]), 0))
    max_y = int(max(np.max(corners_transformed[:, :, 1]), h5))

    # Translation matrix to shift the coordinates to positive space
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # Warp img1 and create an output canvas large enough to hold both images
    stitched_shape = (max_y - min_y, max_x - min_x)
    stitched_img = cv2.warpPerspective(img1, translation.dot(H), stitched_shape)

    # Ensure img5 fits within the bounds of the stitched image
    y_end = min(-min_y + h5, stitched_img.shape[0])
    x_end = min(-min_x + w5, stitched_img.shape[1])

    # Copy img5 into the corresponding region of stitched_img
    stitched_img[-min_y:y_end, -min_x:x_end] = img5[:y_end + min_y, :x_end + min_x]

    # Show the final stitched result
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB))
    plt.title('Stitched Image')
    plt.axis('off')
    plt.show()

    # Save results
    cv2.imwrite('stitched_image.jpg', stitched_img)
    print("\nStitched image saved as 'stitched_image.jpg'.")

    return H

if __name__ == "__main__":
    img1_path = "data/graf/img1.ppm"
    img5_path = "data/graf/img5.ppm"
    
    # Load the ground truth homography matrix
    file_path = "data/graf/H1to5p"
    with open(file_path, 'r') as f:
        data = f.read()
    gt_H = np.array([float(x) for x in data.split()]).reshape(3, 3)

    # Run the stitching pipeline
    try:
        computed_H = sift_and_stitch(img1_path, img5_path, gt_homography=gt_H)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
