# Face Recognition Web Application

A modern web-based face recognition system built with Flask and InsightFace. This application provides an intuitive interface for enrolling students and performing real-time face recognition.

## 🌟 Features

### Original CLI Tools (Preserved)
- **enrollment.py** - Command-line enrollment with multiple samples
- **recognize.py** - Command-line live recognition
- **utils.py** - Core utility functions

### New Web Application
- **Modern Web Interface** - Clean, responsive design
- **Student Enrollment** - Single-shot or multiple-sample enrollment
- **Live Recognition** - Real-time face verification
- **Student Management** - View and delete enrolled students
- **Adjustable Settings** - Configure threshold and recognition interval
- **Visual Feedback** - Real-time status indicators and overlays

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**
```bash
cd /home/pankaja/Projects/face-recognition-insightface
```

2. **Install dependencies for web app**
```bash
pip install -r web_requirements.txt
```

Or use the original requirements:
```bash
pip install -r requirements.txt
pip install Flask Pillow
```

### Running the Web Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

### Using Original CLI Tools

**Enrollment:**
```bash
python enrollment.py
```

**Recognition:**
```bash
python recognize.py
```

## 📖 Web App Usage

### 1. Home Page
- View enrolled student count
- Navigate to enrollment or recognition

### 2. Enrollment Page
- **Single Shot Mode**: Quick enrollment with one image
- **Multiple Samples Mode**: Better accuracy with 5 images
- Enter student ID and capture face images
- Visual feedback for captured samples

### 3. Recognition Page
- Start/stop live recognition
- Adjust recognition threshold (stricter or lenient)
- Set recognition interval
- View enrolled students
- Delete students if needed
- Real-time status indicators

## 🎯 Tips for Best Results

### Enrollment
- Ensure good lighting on face
- Remove glasses if possible
- Look directly at camera
- Neutral facial expression
- No other people in frame
- For multiple samples: vary angles slightly

### Recognition
- Face the camera directly
- Stay within frame boundaries
- Maintain consistent lighting
- Only one person in frame at a time

## 🛠️ Configuration

### Recognition Threshold
- **Default**: 0.40
- **Lower** (0.2-0.35): Stricter matching, fewer false positives
- **Higher** (0.45-0.7): More lenient, may accept similar faces

### GPU Acceleration
To use GPU (if available), edit `app.py`:
```python
initialize_model(device='cuda')  # Change from 'cpu' to 'cuda'
```

## 📁 Project Structure

```
face-recognition-insightface/
├── app.py                      # Flask web application
├── enrollment.py               # Original CLI enrollment (unchanged)
├── recognize.py                # Original CLI recognition (unchanged)
├── utils.py                    # Original utilities (unchanged)
├── requirements.txt            # Original requirements
├── web_requirements.txt        # Web app requirements
├── templates/
│   ├── index.html             # Home page
│   ├── enroll.html            # Enrollment page
│   └── recognize.html         # Recognition page
├── static/
│   └── style.css              # Stylesheet
└── student_embeddings.pkl     # Stored face embeddings
```

## 🔧 API Endpoints

- `GET /` - Home page
- `GET /enroll` - Enrollment page
- `GET /recognize` - Recognition page
- `POST /api/enroll` - Enroll single image
- `POST /api/enroll_multiple` - Enroll multiple samples
- `POST /api/recognize` - Recognize face
- `GET /api/students` - List enrolled students
- `DELETE /api/delete_student/<id>` - Delete student

## 🔒 Security Notes

For production deployment:
1. Change the Flask secret key in `app.py`
2. Disable debug mode
3. Use proper authentication
4. Implement rate limiting
5. Use HTTPS

## 📝 Original Files

All original files remain unchanged:
- `enrollment.py` - Original enrollment script
- `recognize.py` - Original recognition script
- `utils.py` - Original utility functions
- `requirements.txt` - Original dependencies

## 🤝 Technologies Used

- **Flask** - Web framework
- **InsightFace** - Face recognition model
- **OpenCV** - Image processing
- **JavaScript** - Frontend interactivity
- **HTML/CSS** - User interface

## 📄 License

Same as original project

---

**Note**: The original command-line tools (`enrollment.py` and `recognize.py`) remain fully functional and unchanged. The web application is an additional interface built on top of the same core utilities.
