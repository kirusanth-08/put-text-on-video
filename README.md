# Video Caption API

A FastAPI-based web service that adds text captions to videos and uploads them to Cloudinary. This application allows you to overlay customizable text captions on video files with various font options and sizes.

## Features

- Process video files to add text captions
- Customizable font styles and sizes
- Automatic text wrapping and positioning
- Cloud storage integration with Cloudinary
- RESTful API endpoints
- CORS support for cross-origin requests

## Prerequisites

- Python 3.8+
- OpenCV
- FastAPI
- Cloudinary account
- FFmpeg (for video processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video-caption-api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your Cloudinary credentials:
```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
CLOUDINARY_UPLOAD_PRESET=your_upload_preset
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /process-video

Processes a video from a URL by adding captions and uploads it to Cloudinary.

**Request Body (JSON):**
```json
{
    "video_url": "https://example.com/video.mp4",
    "caption_text": "Your caption text here",
    "font_choice": "FONT_HERSHEY_SIMPLEX",
    "font_size": 1.0
}
```

- `video_url`: URL to the video file as string
- `caption_text`: Text to overlay on the video
- `font_choice`: OpenCV font type (optional, default: "FONT_HERSHEY_SIMPLEX")
- `font_size`: Font size (optional, default: 1.0)

**Available Font Choices:**
- FONT_HERSHEY_SIMPLEX
- FONT_HERSHEY_PLAIN
- FONT_HERSHEY_DUPLEX
- FONT_HERSHEY_COMPLEX
- FONT_HERSHEY_TRIPLEX
- FONT_HERSHEY_COMPLEX_SMALL
- FONT_HERSHEY_SCRIPT_SIMPLEX
- FONT_HERSHEY_SCRIPT_COMPLEX

**Response:**
```json
{
    "status": "success",
    "video_url": "https://cloudinary.com/...",
    "public_id": "video_captions/..."
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- 400: Bad Request (invalid video file)
- 500: Internal Server Error (processing/upload failed)

## Development

The application uses:
- FastAPI for the web framework
- OpenCV for video processing
- Cloudinary for cloud storage
- python-multipart for handling file uploads
- python-dotenv for environment variables

## API Documentation

FastAPI provides automatic API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security Notes

- Keep your `.env` file secure and never commit it to version control
- Use appropriate CORS settings in production
- Consider implementing rate limiting for production use
- Monitor Cloudinary usage to manage costs

## Support

For support, please open an issue in the GitHub repository.


