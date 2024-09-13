from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

app = FastAPI(
    title="Object Detection FastAPI Template",
)

# Initialize the models
model = YOLO("model/best.pt")

@app.post('/predict')
def predict(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    np_array = np.array(input_image)

    output = model.predict(np_array, conf=0.4, iou=0.6,imgsz=704, half=True, device="0")  # results list
    # print(output)

    for index in range(len(output[0].boxes)):
        pts = output[0].masks[index].xy[0].astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
    
        isClosed = True
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
        np_array = cv2.polylines(np_array, [pts],
                            isClosed, color, thickness)
    
    image = Image.fromarray(np_array) 
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Save as PNG or any other format
    img_byte_arr.seek(0)

    return StreamingResponse(content=img_byte_arr, media_type="image/jpeg")

    
    
