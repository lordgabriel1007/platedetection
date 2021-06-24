import cv2
import numpy as np

def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if (area > 1000.0) & (area < 5000.0):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    warped = None  # Stores the warped license plate image
    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) # Source points
        height = 160 #image.shape[0]
        width = 360 #image.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, (img_shape), flags=cv2.INTER_LINEAR)

    return biggest, imgContour, warped  # Change - also return drawn image


image = cv2.imread('aia3326-small.png')
cv2.imshow("orig", image)
cv2.waitKey(0)
kernel = np.ones((3,3))
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", imgGray)
cv2.waitKey(0)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
cv2.imshow("blur", imgBlur)
cv2.waitKey(0)

imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
cv2.imshow("thresh", imgThresh)
cv2.waitKey(0)

'''imgCanny = cv2.Canny(imgThresh,150,200)
cv2.imshow("canny", imgCanny)
cv2.waitKey(0)
imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
cv2.imshow("dilate", imgDial)
cv2.waitKey(0)

imgThres = cv2.erode(imgDial,kernel,iterations=2)
cv2.imshow("dilate", imgThres)
cv2.waitKey(0)'''

biggest, imgContour, warped = getContours(imgThresh, image)

cv2.imshow("contour", imgContour)
cv2.waitKey(0)