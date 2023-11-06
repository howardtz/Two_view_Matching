import cv2

img_read1 = cv2.imread('two-view matching/00000022.jpg')
img_read2 = cv2.imread('two-view matching/00000024.jpg')


sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img_read1, None)
kp2, des2 = sift.detectAndCompute(img_read2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img_read1, kp1, img_read2, kp2, matches[:200], None, flags=2)

img_write = cv2.imwrite('sift_keypoints.jpg', img3)