import cv2
import numpy as np

img1 = cv2.imread("./data/00000022.jpg")
img2 = cv2.imread("./data/00000023.jpg")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Step 1: feature extraction
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_gray,None)
kp2, des2 = sift.detectAndCompute(img2_gray,None)

# Step 2: feature match
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)

pts1 = []
pts2 = []
# ratio test as per Lowe's paper
k = 0
goodMatch = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.45 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        goodMatch.append(m)
        k = k + 1
        if k == 8:
            # only reserve 8 points
            break
# threshold = 100
# for match in matches:
#     if match.distance < threshold:
#         goodMatch.append(match)
#         pts2.append(kp2[match.trainIdx].pt)
#         pts1.append(kp1[match.queryIdx].pt)
#         k = k + 1
#         if k == 8:
#             # only reserve 8 points
#             break
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


# opencv's default Fundamental Matrix estimation
F, mask = cv2.findFundamentalMat(pts1,pts2)
print(F)

p_a = np.array([[pts1[1][0]], [pts1[1][1]], [1]])
p_b = np.array([[pts2[1][0]], [pts2[1][1]], [1]])
r_ocv = np.matmul(p_a.T, F)
r_ocv = np.matmul(r_ocv, p_b)
print(r_ocv)

def generate_fundamental_matrix(pt1_list, pt2_list):
    A = []
    for i in range(len(pt1_list)):
        x_a = pt1_list[i,0]
        y_a = pt1_list[i,1]
        x_b = pt2_list[i,0]
        y_b = pt2_list[i,1]
        row = [x_a * x_b, x_b * y_a, x_b,
               y_b * x_a, y_a * y_b, y_b,
                     x_a,       y_a,   1]
        A.append(row)
    A = np.array(A)
    U, D, V = np.linalg.svd(A, full_matrices=True)
    F = V[-1, :]
    F = np.reshape(F, (3, 3))

    U, D, V = np.linalg.svd(F, full_matrices=True)
    d = np.zeros((3, 3))
    d[0, 0] = D[0]
    d[1, 1] = D[1]
    F = np.dot(U, np.dot(d, V))
    return F


F_svd = generate_fundamental_matrix(pts1,pts2)
print(F_svd)

p_a = np.array([[pts1[1][0]], [pts1[1][1]], [1]])
p_b = np.array([[pts2[1][0]], [pts2[1][1]], [1]])
r_ocv = np.matmul(p_a.T, F_svd)
r_ocv = np.matmul(r_ocv, p_b)
print(r_ocv)

res = cv2.drawMatches(img1,kp1,img2,kp2,goodMatch,None,(0,255,0))

cv2.imwrite('res.jpg',res)