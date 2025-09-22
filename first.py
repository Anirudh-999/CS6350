import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 1")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title("Image 2")
plt.axis('off')
plt.suptitle("STEP 1: Loaded Images")
plt.show()

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img1_kp)
plt.title("Image 1 Keypoints")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img2_kp)
plt.title("Image 2 Keypoints")
plt.axis('off')
plt.suptitle("STEP 2: SIFT Keypoints")
plt.show()

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches found:", len(good_matches))

good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 6))
plt.imshow(img_matches)
plt.title("STEP 3: Top 50 Matches")
plt.axis('off')
plt.show()

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

pts1_in = pts1[mask.ravel() == 1]
pts2_in = pts2[mask.ravel() == 1]

print("Inlier matches after RANSAC:", len(pts1_in))

inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
img_inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 6))
plt.imshow(img_inlier_matches)
plt.title("STEP 4: Inlier Matches after RANSAC")
plt.axis('off')
plt.show()

K = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)

E = K.T @ F @ K

plt.figure()
plt.imshow(F, cmap='bwr')
plt.colorbar()
plt.title("STEP 5: Fundamental Matrix Visualization")
plt.show()

_, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)

print("Estimated Rotation:\n", R)
print("Estimated Translation:\n", t)

def draw_epipolar_lines(img1, img2, F, pts1, pts2, num_lines=10):
    pts1 = pts1[:num_lines]
    pts2 = pts2[:num_lines]
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [img1.shape[1], -(r[2]+r[0]*img1.shape[1])/r[1] ])
        img1_color = cv2.line(img1_color, (x0,y0), (x1,y1), color,1)
        img1_color = cv2.circle(img1_color, tuple(np.int32(pt1)),5,color,-1)
        img2_color = cv2.circle(img2_color, tuple(np.int32(pt2)),5,color,-1)
    return img1_color, img2_color

img1_epi, img2_epi = draw_epipolar_lines(img1, img2, F, pts1_in, pts2_in)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img1_epi)
plt.title("STEP 6: Epipolar Lines (Image 1)")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img2_epi)
plt.title("STEP 6: Epipolar Points (Image 2)")
plt.axis('off')
plt.show()

proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
proj2 = K @ np.hstack((R, t))

pts1_h = cv2.convertPointsToHomogeneous(pts1_in)[:, 0, :]
pts2_h = cv2.convertPointsToHomogeneous(pts2_in)[:, 0, :]

pts_4d_h = cv2.triangulatePoints(proj1, proj2, pts1_in.T, pts2_in.T)
pts_3d = (pts_4d_h / pts_4d_h[3])[:3].T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2], s=10)
ax.set_title("STEP 7: Triangulated 3D Points")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

avg_depth = np.mean(pts_3d[:, 2])
theta = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi

print("Average Depth:", avg_depth)
print("Average Pose Change (angle):", theta, "degrees")

plt.figure()
plt.text(0.1, 0.7, f"STEP 8: Avg Depth: {avg_depth:.2f}\nPose Change: {theta:.2f} deg", fontsize=16)
plt.axis('off')
plt.show()
