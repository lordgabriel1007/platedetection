import cv2
import pytesseract
import pandas as pd
import numpy as np
import argparse
import time
from datetime import datetime
import os
import glob
import re
import easyocr
import serial
from ultralytics import YOLO

reader = easyocr.Reader(['en'], gpu=False)
model = YOLO('license_plate_detector.pt')

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        return text, score

print("main program starting..")

# Initialize serial communication with Arduino
ser = None
try:
    ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the appropriate serial port
except:
    print("Could not connect to Arduino - gate control. Will proceed without gate control.")

def gate_control(action):
    if ser is not None:
        if action == "OPEN":
            ser.write(b'o')
        elif action == "CLOSE":
            ser.write(b'c')
        elif action == "RED":
            ser.write(b'r')
        elif action == "OPENGREEN":
            ser.write(b'og')
        elif action == "OFF":
            ser.write(b'x')



def process(frame):
    # Perform object detection on the frame
    results = model.predict(frame, verbose=False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    plate_text = ""
    score = 0.0
    x_min = 0
    y_min = 0
    cropped_image = None
    plate_saved = False
    # Iterate through the detected objects
    for result in results:
        if result.boxes:
            
            # Iterate through all detected objects in the current frame
            for box in result.boxes:
            
                # Get the bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                # box_width = x_max - x_min
                confidence = box.conf[0]

                if confidence > 0.7 and plate_saved==False:
                    # Crop the image
                    cropped_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            
                     # Save the cropped image
                    # cv2.imwrite(f'cropped_image_{i}.png', cropped_image)

                    plate_saved = True
                    # Break the loop after processing the first detection

                    plate_text, score = read_license_plate(cropped_image)
                    print(f'DETECTED:{plate_text} : SCORE:{score}')
                    break

    return plate_text.strip().upper(), score,  x_min, y_min, annotated_frame, cropped_image



def video_processor(vid_capture, df, source):
    frame_rate = 30
    # initialize state variables
    prev = 0
    frame_no = 0
    state = 1  # initial state, no car visible
    text = ""
    direction = ""
    car_type = ""
    found = False
    gate = "closed"
    while vid_capture.isOpened():
        time_elapsed = time.time() - prev
        # Capture each frame of webcam video
        ret, image = vid_capture.read()
        if time_elapsed > 1.0 / frame_rate:
            prev = time.time()
            text, score,  x_min, y_min, annotated_frame, cropped_image = process(image)
            # cv2.imshow("annotated_frame", annotated_frame)
            if type(cropped_image) != type(None):
                cv2.imshow("plate image", cropped_image)
                print(f"detected text:{text}, length:{len(text)}")
                text = pattern.sub("", text)  # keep alphanumeric characters only
                # print(f"frame #{frame_no}")
                # implement state machine here:
                if state == 1:  # no car plate visible
                    x = re.search("^[A-Z]{3}[0-9]{3,4}$", text)
                    # check string is a plate number
                    if x:
                        frame_begin = frame_no  # save the frame number
                        corner_x = x_min  # save the x coordinate of the upper left corner
                        plate_no = text  # save text as plate number
                        print(f"plate=[{plate_no}] found at frame#{frame_no}")
                        found = True
                        text = ""  # reset text variable
                        state = 2  # move to state 2
                        # check if car is resident or visitor
                        if len(rdf[rdf.Plate == plate_no]) == 0:
                            car_type = "VISITOR"
                            gate_control("RED")
                        else:
                            car_type = "RESIDENT"
                            # OPEN THE GATE
                            gate_control("OPENGREEN")
                            gate = "open"
                elif state == 2:
                    x = re.search("^[A-Z]{3}[0-9]{3,4}$", text)
                    if x:
                        frame_begin = frame_no
                        # print(f'corners: {corners}')
                        if x_min > 0:
                            delta_x = x_min - corner_x
                            if delta_x > 0:
                                direction = "EXITING"
                                print("moving left to right")
                            else:
                                direction = "ENTERING"
                                print("moving right to left")
                            # use this current one as it is closer to the camera
                            plate_no = text  # save text as plate number
                            frame_elapsed = frame_no - frame_begin
                            if frame_elapsed > 40:
                                # append this car to the dataframe
                                new_row = {
                                    "DateTime": datetime.now(),
                                    "Plate": plate_no,
                                    "Direction": direction,
                                    "Type": car_type,
                                    "Source": source,
                                }
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                                # next state = 1
                                direction = ""
                                car_type = ""
                                plate_no = ""
                                # CLOSE THE GATE
                                gate_control("CLOSE")
                                gate = "closed"
                                state = 1
                                text = ""
            frame_no += 1
            print(f"state={state}")
            # output.write(frame)
            # Close and break the loop after pressing "ESC" key
            keypressed = cv2.waitKey(1)
            if keypressed == 27:
                break
            elif keypressed & 0xFF == ord("o"):
                gate = "open"
                gate_control("OPEN")
            elif keypressed & 0xFF == ord("c"):
                gate = "closed"
                gate_control("CLOSE")
            if source != "webcam":
                if vid_capture.get(cv2.CAP_PROP_POS_FRAMES) == vid_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                    # If the number of captured frames is equal to the total number of frames,
                    # we stop
                    break
    # if state machine ends at state 2, append the last recorded car
    if state == 2:
        new_row = {
            "DateTime": datetime.now(),
            "Plate": plate_no,
            "Direction": direction,
            "Type": car_type,
            "Source": source,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    if found == False:
        new_row = {
            "DateTime": datetime.now(),
            "Plate": "",
            "Direction": "",
            "Type": "",
            "Source": source,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    if gate == "open":
        # CLOSE THE GATE
        gate_control("CLOSE")
        gate = "closed"
    # close the already opened camera
    vid_capture.release()
    # close the already opened file
    # output.release()
    # close the window and de-allocate any associated memory usage
    cv2.destroyAllWindows()
    return df

# PROGRAM START HERE
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--videofolder", type=str, default="", help="video folder to load from")
args = vars(ap.parse_args())

# Load residents' plate list
rdf = pd.read_excel("residents.xlsx")

# initialize data frame to store detected plates
df = pd.DataFrame(columns=["DateTime", "Plate", "Direction", "Type", "Source"])

# Regular expression for stripping out non-alphanumeric characters
pattern = re.compile("[\W_]+")

if args["videofolder"] == "":
    # Capture video from webcam
    vid_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    source = "webcam"
    video_processor(vid_cap, df, source)
else:
    all_files = glob.glob(os.path.join(f'{args["videofolder"]}/*.mp4'))
    for filename in all_files:
        print(filename)
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        vid_cap = cv2.VideoCapture(filename)
        source = os.path.basename(filename)
        # Check if camera opened successfully
        if vid_cap.isOpened() == False:
            print(f"Error opening {filename}")
            continue
        # process the opened video file
        df = video_processor(vid_cap, df, source)

today = str.replace(f"data-{datetime.now()}", ":", "_")
df.to_excel(f"{today}.xlsx")
gate_control("OFF")

if ser is not None:
    ser.close()