from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
app = FastAPI()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Body Measurement API"}

@app.post("/measure")
async def measure(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        return {"error": "No person detected"}
    
    # Extract body measurements (example: height of a person)
    height_pixels = get_body_height(results.pose_landmarks.landmark, img.shape[0])
    
    # Generate a basic 3D avatar (not implemented in this example)
    avatar_img = generate_avatar_image()
    avatar_b64 = image_to_base64(avatar_img)
    
    return {"height_pixels": height_pixels, "avatar": avatar_b64}

def get_body_height(landmarks, image_height):
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    height_pixels = abs(ankle.y - shoulder.y) * image_height
    return height_pixels

def generate_avatar_image():
    # Create a simple image with PIL for demonstration purposes
    img = Image.new('RGB', (100, 200), color = (73, 109, 137))
    return img

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
