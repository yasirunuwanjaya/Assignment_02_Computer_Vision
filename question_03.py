import cv2 as cv
import numpy as np
import sys

# Load images
im1 = cv.imread('c1.jpg', cv.IMREAD_REDUCED_COLOR_4)
im2 = cv.imread('c2.jpg', cv.IMREAD_REDUCED_COLOR_4)

if im1 is None or im2 is None:
    print("Error: Images not found in 'a2_images/' or current folder.")
    sys.exit()

N = 6
n = 0
p1 = np.zeros((N, 2), dtype=np.float32)
p2 = np.zeros((N, 2), dtype=np.float32)

def draw_circle(event, x, y, flags, param):
    global n
    p, img, win = param
    if event == cv.EVENT_LBUTTONDOWN and n < N:
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        p[n] = (x, y)
        n += 1
        cv.imshow(win, img)
        print(f"Point {n}/{N} recorded.")

# --- Manual Selection ---
for i, (pts, image, name) in enumerate([(p1, im1, "Image 1"), (p2, im2, "Image 2")]):
    n = 0
    img_copy = image.copy()
    cv.namedWindow(name)
    cv.setMouseCallback(name, draw_circle, [pts, img_copy, name])
    print(f"Select {N} points on {name}. Press 'ESC' to abort.")
    while n < N:
        cv.imshow(name, img_copy)
        if cv.waitKey(1) & 0xFF == 27: sys.exit()
    cv.destroyWindow(name)

# --- Calculations (Manual) ---
H_manual, _ = cv.findHomography(p1, p2, cv.RANSAC)
im1_warped_man = cv.warpPerspective(im1, H_manual, (im2.shape[1], im2.shape[0]))
diff_manual = cv.absdiff(im2, im1_warped_man)

# --- Automated (ORB) ---
orb = cv.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(im1, None)
kp2, des2 = orb.detectAndCompute(im2, None)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H_auto, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
im1_warped_auto = cv.warpPerspective(im1, H_auto, (im2.shape[1], im2.shape[0]))
diff_auto = cv.absdiff(im2, im1_warped_auto)

# --- Display Final Comparison ---
cv.imshow("Manual Difference", diff_manual)
cv.imshow("Auto (ORB) Difference", diff_auto)
cv.waitKey(0)
cv.destroyAllWindows()