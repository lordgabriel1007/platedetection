import cv2


from imutils.object_detection import non_max_suppression
import pytesseract
import numpy as np
import argparse
import time

print("main_ocr starting..")


def union(a, b):
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])

    x2 = max(a[2], b[2])
    y2 = max(a[3], b[3])
    return (x1, y1, x2, y2)


def process(image):
    # step 1, dewarp

    # pts1 = np.float32([[445, 494], [977, 406], [492, 944], [994,cls 802]])
    # pts2 = np.float32([[0, 0], [320, 0], [0, 240], [320, 240]])

    pts1 = np.float32([[538, 19], [950, 115], [580, 518], [985, 660]])
    pts2 = np.float32([[0, 0], [320, 0], [0, 240], [320, 240]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (320, 240))

    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    print(f"rectangles: {len(rects)}")

    # if no text found
    if len(rects) == 0:
        return orig.copy()

    # print(f"number of rects: {len(rects)}")
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    print(f"number of boxes: {len(boxes)}")

    if len(boxes) > 1 & len(boxes) < 3:
        bigbox = boxes[0]
        for x in range(1, len(boxes)):
            bigbox = union(bigbox, boxes[x])
        boxes = [bigbox]

    # convert to gray
    # orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * args["padding"])
        dY = int((endY - startY) * args["padding"])

        dX = 0

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = "-l eng --oem 1 --psm 7"
        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])

    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")
        print("{}".format(text))

        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            output,
            text,
            (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
    return output


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


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

layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
print (args['east'])
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

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


frame_rate = 5
prev = 0
frame_no = 0
while vid_capture.isOpened():
    time_elapsed = time.time() - prev
    # Capture each frame of webcam video
    ret, image = vid_capture.read()

    if time_elapsed > 1.0 / frame_rate:
        prev = time.time()

        output = process(image)

        cv2.imshow("My cam video", output)
        print(f"frame #{frame_no}")
        frame_no += 1

    # output.write(frame)
    # Close and break the loop after pressing "x" key
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

# close the already opened camera
vid_capture.release()
# close the already opened file
# output.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()
