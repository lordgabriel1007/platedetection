import ollama
from ollama import generate
import cv2

def process_image(image_bytes):
    full_response = ''
    # Generate a description of the image
    for response in generate(model='llava:13b-v1.6', 
                             prompt='what type of vehicle is in this image? answer in this format[manufacturer vehicle_type] examples: [Mitsubishi Sedan], [Toyota SUV], [Ford Pickup] ', 
                             images=[image_bytes], 
                             stream=True):
        # Print the response to the console and add it to the full response
        print(response['response'], end='', flush=True)
        full_response += response['response']
    return full_response
        

img = cv2.imread('aia3326-small.png')
is_success, buffer = cv2.imencode(".png", img)
io_buf = buffer.tobytes()

analysis = process_image(io_buf)

