import cv2


from imutils.object_detection import non_max_suppression
import pytesseract
import numpy as np
import argparse
import time

print("main program starting..")


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
        if (area > 5000.0) & (area < 20000.0):
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


def process(image):
    kernel = np.ones((3,3))
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,150,200)
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=2)
    biggest, imgContour, warped = getContours(imgThres, image)  # Change
    return imgContour, warped

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
ap.add_argument(
    "-c",
    "--min-confidence",
    type=float,
    default=0.5,
    help="minimum probability required to inspect a region",
)
ap.add_argument(
    "-w",
    "--width",
    type=int,
    default=320,
    help="nearest multiple of 32 for resized width",
)
ap.add_argument(
    "-e",
    "--height",
    type=int,
    default=320,
    help="nearest multiple of 32 for resized height",
)
ap.add_argument(
    "-p",
    "--padding",
    type=float,
    default=0.0,
    help="amount of padding to add to each border of ROI",
)
ap.add_argument(
    "-v",
    "--videofile",
    type=str,
    default="",
    help="video file to load",
)
args = vars(ap.parse_args())



if args["videofile"] == "":
    # Capture video from webcam
    vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vid_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
else:
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    vid_capture = cv2.VideoCapture(args["videofile"])

    # Check if camera opened successfully
    if vid_capture.isOpened() == False:
        print("Error opening video stream or file")
        exit()


# vid_cod = cv2.VideoWriter_fourcc(*"MJPG")
# output = cv2.VideoWriter("cam_video.mp4", vid_cod, 20.0, (640, 480))


frame_rate = 10
prev = 0
frame_no = 0

state = 1 #no car
text = ''
while vid_capture.isOpened():
    time_elapsed = time.time() - prev
    # Capture each frame of webcam video
    ret, image = vid_capture.read()

    if time_elapsed > 1.0 / frame_rate:
        prev = time.time()

        contour, warped = process(image)

        cv2.imshow("contour", contour)
        if (type(warped) != type(None)):
            cv2.imshow("warped", warped)
            config = "-l eng --oem 1 --psm 7"
            text = pytesseract.image_to_string(warped, config=config)
            print(f'detected text:{text}, length:{len(text)}')



        print(f"frame #{frame_no}")
        frame_no += 1

        #implement state machine here:
        if state==1: #no car plate visible
            #check string length
            if (len(text)>= 6):
                state = 2 #first view of plate
                frame_begin = frame_no
                print(f'plate found at frame#{frame_no}')
        elif state==2: 
            if (len(text)>= 6):
                frame_begin = frame_no
            frame_elapsed = frame_no - frame_begin
            if frame_elapsed > 10:
                state = 1
            
        print(f'state={state}')

    # output.write(frame)
    # Close and break the loop after pressing "ESC" key
    if cv2.waitKey(1) == 27:
        break

    if args["videofile"] != "":
        if vid_capture.get(cv2.CAP_PROP_POS_FRAMES) == vid_capture.get(cv2.CAP_PROP_FRAME_COUNT) :
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

# close the already opened camera
vid_capture.release()
# close the already opened file
# output.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()
