import cv2
import os
import numpy as np
import time
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from typing import List
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

def wrap_text(text, font_scale, thickness, max_width, font):
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_size[0] > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    
    lines.append(current_line)
    return lines

def add_text_captions(video_path: str, caption_text: str, font_choice: str, font_size: float, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Error: Cannot open video file {video_path}")
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    margin = int(frame_width * 0.05)
    max_text_width = frame_width - 2 * margin
    font_scale = float(font_size)
    thickness = 2
    font = getattr(cv2, font_choice)
    
    while True:
        wrapped_lines = wrap_text(caption_text, font_scale, thickness, max_text_width, font)
        text_height = len(wrapped_lines) * cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 10
        if text_height < frame_height * 0.2:
            break
        font_scale -= 0.05
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        text_y = frame_height - 50 - text_height
        
        for line in wrapped_lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (frame_width - text_size[0]) // 2
            cv2.putText(frame, line, (text_x, text_y), font, 
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            text_y += text_size[1] + 10
        
        out.write(frame)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

@app.post("/process-video")
async def process_video(
    video: UploadFile,
    caption_text: str = Form(...),
    font_choice: str = Form("FONT_HERSHEY_SIMPLEX"),
    font_size: float = Form(1.0)
):
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            temp_video_path = os.path.join(temp_dir, video.filename)
            with open(temp_video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            # Generate output path
            timestamp = int(time.time())
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Process video
            processed_video_path = add_text_captions(
                temp_video_path,
                caption_text,
                font_choice,
                font_size,
                output_path
            )
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                processed_video_path,
                resource_type="video",
                folder="video_captions",
                upload_preset=os.getenv('CLOUDINARY_UPLOAD_PRESET')
            )
            
            return {
                "status": "success",
                "video_url": upload_result["secure_url"],
                "public_id": upload_result["public_id"]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
