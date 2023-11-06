import cv2
import random
import numpy as np

def fundamental_matrix(pts1, pts2):
    A = []
    for i in range(len(pts1)):
        p1_x = pts1[i,0]
        p1_y = pts1[i,1]
        p2_x = pts2[i,0]
        p2_y = pts2[i,1]
        row = [p1_x * p2_x, p2_x * p1_y, p2_x, p2_y * p1_x, p1_y * p2_y, p2_y, p1_x, p1_y, 1]
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

def normalized_fundamental_matrix(pts1, pts2):
    mean_p1_x = np.mean(pts1[:, 0])
    mean_p1_y = np.mean(pts1[:, 1])
    mean_p2_x = np.mean(pts2[:, 0])
    mean_p2_y = np.mean(pts2[:, 1])
    sx1 = 1 / np.mean(np.abs(pts1[:, 0] - mean_p1_x))
    sy1 = 1 / np.mean(np.abs(pts1[:, 1] - mean_p1_y))
    sx2 = 1 / np.mean(np.abs(pts2[:, 0] - mean_p2_x))
    sy2 = 1 / np.mean(np.abs(pts2[:, 1] - mean_p2_y))
    pts1_norm = np.array([[sx1 * (x - mean_p1_x), sy1 * (y - mean_p1_y)] for x, y in pts1])
    pts2_norm = np.array([[sx2 * (x - mean_p2_x), sy2 * (y - mean_p2_y)] for x, y in pts2])
    T1 = np.array([[sx1, 0, -sx1 * mean_p1_x],
                   [0, sy1, -sy1 * mean_p1_y],
                   [0, 0, 1]])
    T2 = np.array([[sx2, 0, -sx2 * mean_p2_x],
                   [0, sy2, -sy2 * mean_p2_y],
                   [0, 0, 1]])
    F = fundamental_matrix(pts1_norm, pts2_norm)
    F = np.dot(T1.T, np.dot(F, T2))
    return F

def RANSAC_8points(good_matches):
    pts1 = []
    pts2 = []
    points = random.sample(good_matches, 8)
    for point in points:
        pts1.append(kp1[point.queryIdx].pt)
        pts2.append(kp2[point.trainIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    return pts1,pts2,points

def f_judge(F,good_matches,kps1,kps2,threshold):
    num = 0
    ransac_good = []
    for good_match in good_matches:
        p1 = kps1[good_match.queryIdx].pt
        p2 = kps2[good_match.trainIdx].pt
        p1 = np.array([p1[0], p1[1], 1])
        p2 = np.array([p2[0], p2[1], 1])
        res = np.matmul(np.matmul(p1.T, F), p2)
        if abs(res) < threshold:
            ransac_good.append(good_match)
    return ransac_good

def RANSAC_8point_F(good_matches,kps1,kps2,threshold,acc):
    while True:
        pts1,pts2,points = RANSAC_8points(good_matches)
        F = normalized_fundamental_matrix(pts1,pts2)
        ransac_good = f_judge(F,good_matches,kps1,kps2,threshold)
        if len(ransac_good) >= acc * len(good_matches):
            return F, ransac_good, points

def main():
    img1 = cv2.imread("./data/00000022.jpg")
    img2 = cv2.imread("./data/00000023.jpg")
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # feature extraction
    sift = cv2.SIFT_create()
    # find the key points and descriptors with SIFT
    kps1, des1 = sift.detectAndCompute(img1_gray,None)
    kps2, des2 = sift.detectAndCompute(img2_gray,None)
    # feature match
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # find good match points
    good_matches = []
    for (m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    F, ransac_good, points = RANSAC_8point_F(good_matches,kps1,kps2,0.1,0.95)
    print(F)
    res = cv2.drawMatches(img1,kps1,img2,kps2,points,None)
    cv2.imwrite('res.jpg',res)

if __name__ == '__main__':
    main()