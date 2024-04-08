import cv2
from ultralytics import YOLO
import easyocr
reader = easyocr.Reader(['en'], gpu=False)


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


# Initialize the YOLO model
model = YOLO('license_plate_detector.pt')

# Open the video
video = cv2.VideoCapture('./2k/2024-04-08 16-17-09.mp4')

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize the cropped image
cropped_image = None
i = 0
plate_saved = False
# Process each frame of the video
while True:
    # Read the frame
    ret, frame = video.read()
    # Break the loop if the video is over
    if not ret:
        break

    # Perform object detection on the frame
    results = model.predict(frame, verbose=False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Iterate through the detected objects
    for result in results:
        if result.boxes:
            
            # Iterate through all detected objects in the current frame
            for box in result.boxes:
            
                # Get the bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                box_width = x_max - x_min
                confidence = box.conf[0]

                if confidence > 0.75 and plate_saved==False:
                    # Crop the image
                    cropped_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    i = i + 1
                     # Save the cropped image
                    cv2.imwrite(f'cropped_image_{i}.png', cropped_image)

                    plate_saved = True
                    # Break the loop after processing the first detection

                    plate_text, score = read_license_plate(cropped_image)
                    print(f'DETECTED:{plate_text} : SCORE:{score}')
                    break

    # Check if the cropped image is larger than the previous one
    # if cropped_image is not None and cropped_image.shape[0] * cropped_image.shape[1] > (80 * 80):
   

    # Write the frame to the output video
    # out.write(frame)

    # Display the frame
    # cv2.imshow('frame', frame)
    cv2.imshow('annotated_frame', annotated_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and output objects and close the windows
video.release()
# out.release()
cv2.destroyAllWindows()