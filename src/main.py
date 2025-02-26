import cv2
import os
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import tempfile
import requests
from pydantic import BaseModel, HttpUrl
import logging
from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO)

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

# Add this near the top of the file, after the imports
FONTS_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
FONT_STYLES = {
    'regular': os.path.join(FONTS_DIR, 'regular'),
    'bold': os.path.join(FONTS_DIR, 'bold'),
    'italic': os.path.join(FONTS_DIR, 'italic'),
    'bold-italic': os.path.join(FONTS_DIR, 'bold-italic')
}

AVAILABLE_FONTS = {
    'default': cv2.FONT_HERSHEY_SIMPLEX,
    'arial': 'arial.ttf',
    'roboto': 'roboto.ttf',
    'montserrat': 'montserrat.ttf',
    'proxima': 'proximanova.otf'
}

def get_font_path(font_choice: str, is_bold: bool, is_italic: bool) -> str:
    if font_choice == 'default':
        return cv2.FONT_HERSHEY_SIMPLEX
    
    if font_choice not in AVAILABLE_FONTS:
        raise HTTPException(status_code=400, detail=f"Font '{font_choice}' not available. Choose from: {list(AVAILABLE_FONTS.keys())}")
    
    # Determine the style folder
    style = 'regular'
    if is_bold and is_italic:
        style = 'bold-italic'
    elif is_bold:
        style = 'bold'
    elif is_italic:
        style = 'italic'
    
    # Construct the full font path
    font_file = AVAILABLE_FONTS[font_choice]
    font_path = os.path.join(FONT_STYLES[style], font_file)
    
    # Check if the font file exists
    if not os.path.exists(font_path):
        logging.warning(f"Font style {style} not available for {font_choice}, using regular style")
        font_path = os.path.join(FONT_STYLES['regular'], font_file)
        if not os.path.exists(font_path):
            raise HTTPException(status_code=400, detail=f"Font file not found for {font_choice}")
    
    return font_path

class VideoRequest(BaseModel):
    video_url: str
    caption_text: str
    audio_url: str | None = None
    font_choice: str = "default"  # Now accepts any key from AVAILABLE_FONTS
    font_size: float = 1.0
    vertical_position: float = 100.0  # 0 is top, 100 is bottom
    is_bold: bool = False
    is_italic: bool = False

def download_video(url: str, output_path: str):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading video: {str(e)}")

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

def add_text_captions(video_path: str, caption_text: str, font_choice: str, font_size: float, output_path: str, vertical_position: float = 100.0, is_bold: bool = False, is_italic: bool = False):
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Error: Cannot open video file {video_path}")
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Check if font_choice exists
        if font_choice not in AVAILABLE_FONTS:
            raise HTTPException(status_code=400, detail=f"Font '{font_choice}' not available. Choose from: {list(AVAILABLE_FONTS.keys())}")

        # Validate vertical position
        if vertical_position < 0 or vertical_position > 100:
            raise HTTPException(status_code=400, detail="Vertical position must be between 0 and 100")

        # Handle different font types
        if font_choice == 'default':
            # Use OpenCV's built-in font
            margin = int(frame_width * 0.05)
            max_text_width = frame_width - 2 * margin
            font_scale = float(font_size)
            thickness = 2
            font = AVAILABLE_FONTS[font_choice]
            
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
                
                # Calculate vertical position based on percentage
                usable_height = frame_height - text_height - 100  # 50px margin on top and bottom
                text_y = int((vertical_position / 100.0) * usable_height) + 50  # Add 50px minimum margin
                
                for line in wrapped_lines:
                    text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                    text_x = (frame_width - text_size[0]) // 2
                    cv2.putText(frame, line, (text_x, text_y), font, 
                                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    text_y += text_size[1] + 10
                
                out.write(frame)
        else:
            # Use custom TTF font
            try:
                base_font_size = int(2 * font_size)  # Base size of 60px
                font = ImageFont.truetype(get_font_path(font_choice, is_bold, is_italic), base_font_size)
                
                # Create text overlay once
                overlay = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Calculate text position and wrap
                margin = 50
                max_width = frame_width - 2 * margin
                lines = []
                words = caption_text.split()
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    text_width = draw.textlength(" ".join(current_line), font=font)
                    if text_width > max_width:
                        if len(current_line) > 1:
                            current_line.pop()
                            lines.append(" ".join(current_line))
                            current_line = [word]
                        else:
                            lines.append(" ".join(current_line))
                            current_line = []
                
                if current_line:
                    lines.append(" ".join(current_line))
                
                # Calculate total text height and position
                line_spacing = base_font_size * 0.3
                total_height = len(lines) * (base_font_size + line_spacing)
                
                # Calculate vertical position based on percentage
                usable_height = frame_height - total_height - 100  # 50px margin on top and bottom
                start_y = int((vertical_position / 100.0) * usable_height) + 50  # Add 50px minimum margin
                
                # Draw text with outline
                for i, line in enumerate(lines):
                    text_width = draw.textlength(line, font=font)
                    x = (frame_width - text_width) // 2
                    y = start_y + i * (base_font_size + line_spacing)
                    
                    # Draw outline
                    outline_color = (0, 0, 0)
                    outline_width = 3
                    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                        draw.text((x+dx*outline_width, y+dy*outline_width), line, font=font, fill=outline_color)
                    
                    # Draw main text
                    draw.text((x, y), line, font=font, fill=(255, 255, 255))
                
                overlay_array = np.array(overlay)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Combine frame with text overlay
                    alpha = overlay_array[:, :, 3] / 255.0
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * (1 - alpha) + overlay_array[:, :, c] * alpha
                    
                    out.write(frame)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing custom font: {str(e)}")

        return output_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in add_text_captions: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    return output_path

def merge_audio_with_video(video_path: str, audio_path: str, output_path: str):
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)
        
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=video.fps)
        
        # Clean up
        video.close()
        audio.close()
        final_video.close()
        
        return output_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error merging audio: {str(e)}")

@app.post("/process-video")
async def process_video(request: VideoRequest):
    temp_dir = None
    try:
        logging.info(f"Received request: {request}")
        temp_dir = tempfile.mkdtemp()

        # Download video
        temp_video_path = os.path.join(temp_dir, "input_video.mp4")
        download_video(request.video_url, temp_video_path)
        
        # Process video with captions
        captioned_video_path = os.path.join(temp_dir, "captioned_video.mp4")
        processed_video_path = add_text_captions(
            temp_video_path,
            request.caption_text,
            request.font_choice,
            request.font_size,
            captioned_video_path,
            request.vertical_position,
            request.is_bold,
            request.is_italic
        )
        
        # If audio URL is provided, download and merge audio
        final_video_path = processed_video_path
        if request.audio_url:
            temp_audio_path = os.path.join(temp_dir, "audio.mp3")
            download_video(request.audio_url, temp_audio_path)
            
            final_video_path = os.path.join(temp_dir, "final_video.mp4")
            final_video_path = merge_audio_with_video(
                processed_video_path,
                temp_audio_path,
                final_video_path
            )
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            final_video_path,
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
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary directory and files
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Error cleaning up temporary files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
