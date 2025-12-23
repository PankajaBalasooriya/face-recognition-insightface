# Exam Proctoring Face Recognition System

Face recognition system using InsightFace (ArcFace) for real-time exam proctoring.

## Features

✅ Real-time face detection and recognition  
✅ Multi-face detection alerts  
✅ Unknown person detection  
✅ **Single photo enrollment support** (NEW)  
✅ **Batch enrollment from photo folders** (NEW)  
✅ Student enrollment with robust averaging  
✅ Cosine similarity-based matching  
✅ Configurable similarity threshold  
✅ No cloud dependencies - runs locally  
✅ Optimized for real-time performance  
✅ Privacy-focused (no raw image storage)


## Installation

### 1. Clone or download the system files

Ensure you have these files:
- `enrollment.py`
- `batch_enroll.py`
- `recognize.py`
- `utils.py`
- `requirements.txt`

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**For GPU Acceleration (Optional):**

If you have an NVIDIA GPU with CUDA installed:

```bash
# Uninstall CPU version
pip uninstall onnxruntime

# Install GPU version
pip install onnxruntime-gpu==1.16.0
```

**Requirements:**
- NVIDIA GPU (GTX 1060 or better recommended)
- CUDA Toolkit 11.x or 12.x installed
- cuDNN library

**Verify GPU availability:**
```bash
python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"
```

You should see `CUDAExecutionProvider` in the list if GPU is properly configured.

### 4. Download InsightFace models

On first run, InsightFace will automatically download the required models (~600MB). This is a one-time download.

```bash
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1)"
```

## Usage

### Step 1: Enroll Students

**Option A: Single Photo Enrollment (Recommended - Fastest)**

Run the enrollment script:

```bash
python enrollment.py
```

1. Select **"1. Enroll from photo (single portrait)"**
2. Enter student ID (e.g., `STU001`, `JOHN_DOE`)
3. Provide path to portrait photo (or drag & drop file)
4. System extracts embedding and shows detected face for confirmation

**Photo Requirements:**
- Clear frontal face view
- Good lighting (avoid shadows on face)
- Only one person in the photo (if multiple detected, largest face is used)
- Recommended: Passport-style or ID card photos
- Supported formats: JPG, JPEG, PNG, BMP, TIFF

**Option B: Batch Photo Enrollment (For Multiple Students)**

If you have a folder of portrait photos, use batch enrollment:

```bash
python batch_enroll.py
```

**Folder Structure Example:**
```
student_photos/
├── STU001.jpg
├── STU002.jpg
├── ALICE_SMITH.png
└── BOB_JONES.jpg
```

**How it works:**
1. Place all student portrait photos in a single folder
2. Name each photo with the student ID (e.g., `STU001.jpg`)
3. Run `batch_enroll.py` and provide the folder path
4. System automatically processes all photos and enrolls students

**Features:**
- Processes entire folders automatically
- Filename (without extension) becomes student ID
- Shows progress for each student
- Provides detailed summary report
- Skips already-enrolled students
- GPU acceleration support for faster processing

**Option C: Live Webcam Enrollment (Most Robust)**

For maximum accuracy with webcam capture:

```bash
python enrollment.py
```

1. Select **"2. Enroll from webcam (multiple samples)"**
2. Enter student ID
3. Look at camera - system captures 10 samples automatically
4. Move slightly between captures for better accuracy
5. Wait for confirmation message

**Tips for webcam enrollment:**
- Ensure good lighting
- Face the camera directly
- Remove glasses if possible (or enroll with/without separately)
- Capture at the same distance you'll use during exams
- Enroll each student individually

**Other menu options:**
- `3. List enrolled students` - View all registered students
- `4. Remove student` - Delete a student from the database
- `5. Exit` - Close the enrollment system

### Step 2: Run Live Recognition

Start the proctoring system:

```bash
python recognize.py
```

**Configuration:**
- Default threshold: `0.4` (recommended)
- Lower threshold = stricter matching (fewer false positives)
- Higher threshold = more lenient (may allow more false positives)
- **GPU acceleration**: System automatically detects and offers GPU option if available

**System Statuses:**

| Status | Meaning | Color |
|--------|---------|-------|
| `VERIFIED` | Known student detected | Green |
| `UNKNOWN PERSON` | Face detected but not in database | Red |
| `MULTIPLE FACES DETECTED` | More than one face in frame | Orange |
| `NO FACE DETECTED` | No face visible | Gray |

**Controls:**
- Press `q` - Quit the proctoring system
- Press `r` - Reload student database (after adding new students)


## File Structure

```
exam-proctoring/
├── enrollment.py              # Student enrollment module
├── batch_enroll.py            # Batch photo enrollment 
├── recognize.py               # Live face recognition
├── utils.py                   # Core utilities
├── requirements.txt           # Python dependencies
├── student_embeddings.pkl     # Stored face embeddings
└── README.md                  
```

## Configuration

### Threshold Tuning

Edit `utils.py` to change the default threshold:

```python
DEFAULT_THRESHOLD = 0.4  # Range: 0.0 to 1.0
```

**Recommended thresholds:**
- `0.35-0.40` - High security (recommended for exams)
- `0.40-0.50` - Balanced
- `0.50-0.60` - More lenient

## Troubleshooting

### No Face Detected in Photo

**Solutions:**
1. Ensure face is clearly visible and not obscured
2. Check lighting - avoid backlit photos
3. Use frontal face view (not profile shots)


### Batch Enrollment Failures

**Solutions:**
1. Check all photos meet quality requirements
2. Ensure proper file naming (StudentID.extension)
3. Verify supported formats (JPG, PNG, BMP, TIFF)
4. Check folder path is correct


### Poor Recognition Accuracy

**Solutions:**
1. **Re-enroll with higher quality photos** (most effective)
2. Lower the similarity threshold (e.g., 0.35 instead of 0.4)
3. Ensure consistent lighting between enrollment and recognition
4. For webcam enrollment: Increase samples (`num_samples` in code)
5. Verify camera quality and positioning

## Technical Details

### Architecture
- **Model**: InsightFace Buffalo_L (ArcFace)
- **Embedding**: 512-dimensional normalized vectors
- **Similarity**: Cosine similarity
- **Detection**: SCRFD face detector
- **Recognition**: ArcFace R100

### Embedding Storage
Embeddings are stored in `student_embeddings.pkl`:
```python
{
    "STU001": array([...]),  # 512-D normalized vector
    "STU002": array([...]),
    ...
}
```

## API Reference

### utils.py

```python
cosine_similarity(embedding1, embedding2)
# Returns: float (similarity score)

find_best_match(query_embedding, stored_embeddings, threshold)
# Returns: tuple (student_id, similarity_score)

save_embeddings(embeddings, filepath)
# Saves embedding dictionary to disk

load_embeddings(filepath)
# Loads embedding dictionary from disk

average_embeddings(embeddings_list)
# Returns: averaged and normalized embedding

validate_embedding(embedding)
# Returns: bool (True if valid)
```

### enrollment.py

```python
StudentEnrollment(num_samples=10, device='cpu')
# Initialize enrollment system

enroll_from_photo(student_id, photo_path)
# Enroll student from a single portrait photo
# Returns: bool (success/failure)

capture_face_samples(student_id)
# Capture and store student embeddings from webcam

list_enrolled_students()
# Display all enrolled students

remove_student(student_id)
# Remove student from database
```

### batch_enroll.py

```python
batch_enroll_from_folder(folder_path, device='cpu')
# Enroll multiple students from a folder of photos
# Photo naming convention: StudentID.extension
# Returns: Statistics (successful, failed, skipped)
```

### recognize.py

```python
FaceRecognitionProctor(threshold=0.4, device='cpu')
# Initialize recognition system

verify_frame(frame)
# Returns: (status, face_data_list)

run()
# Start continuous recognition
```

## Quick Start Example

**Enroll 50 students in under 2 minutes:**

```bash
# 1. Collect passport/ID photos and name them by student ID
#    student_photos/STU001.jpg
#    student_photos/STU002.jpg
#    ... etc

# 2. Run batch enrollment
python batch_enroll.py
# Enter folder path: ./student_photos
# Use GPU? y

# 3. Start proctoring
python recognize.py
```

## License

This system uses InsightFace which is licensed under MIT License.

 
**Last Updated**: December 2024  
**Version**: 1.1.0