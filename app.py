"""
Flask Web Application for Face Recognition System
Provides web interface for student enrollment and recognition.
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
from insightface.app import FaceAnalysis
from utils import (
    save_embeddings, 
    load_embeddings, 
    average_embeddings,
    find_best_match,
    validate_embedding,
    DEFAULT_THRESHOLD,
    EMBEDDINGS_FILE
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face-recognition-secret-key'

# Global variables
face_app = None
embeddings_db = {}


def initialize_model(device='cpu'):
    """Initialize InsightFace model."""
    global face_app
    
    print("Initializing InsightFace model...")
    
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1
    
    face_app = FaceAnalysis(
        name='buffalo_l',
        providers=providers
    )
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("Model loaded successfully")


def load_db():
    """Load embeddings database."""
    global embeddings_db
    embeddings_db = load_embeddings(EMBEDDINGS_FILE)
    return len(embeddings_db)


@app.route('/')
def index():
    """Home page."""
    num_students = load_db()
    return render_template('index.html', num_students=num_students)


@app.route('/enroll')
def enroll_page():
    """Enrollment page."""
    return render_template('enroll.html')


@app.route('/recognize')
def recognize_page():
    """Recognition page."""
    num_students = load_db()
    return render_template('recognize.html', num_students=num_students)


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """API endpoint for enrolling a student."""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        image_data = data.get('image')
        
        if not student_id or not image_data:
            return jsonify({'success': False, 'error': 'Missing student_id or image'})
        
        # Check if student already exists
        if student_id in embeddings_db:
            return jsonify({'success': False, 'error': 'Student ID already enrolled'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(frame)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected'})
        
        if len(faces) > 1:
            return jsonify({'success': False, 'error': 'Multiple faces detected. Please ensure only one person is visible.'})
        
        # Extract embedding
        face = faces[0]
        embedding = face.embedding
        
        if not validate_embedding(embedding):
            return jsonify({'success': False, 'error': 'Invalid embedding extracted'})
        
        # Store embedding
        embeddings_db[student_id] = embedding
        save_embeddings(embeddings_db, EMBEDDINGS_FILE)
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} enrolled successfully!'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/enroll_multiple', methods=['POST'])
def api_enroll_multiple():
    """API endpoint for enrolling a student with multiple samples."""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        images = data.get('images', [])
        
        if not student_id or not images:
            return jsonify({'success': False, 'error': 'Missing student_id or images'})
        
        if student_id in embeddings_db:
            return jsonify({'success': False, 'error': 'Student ID already enrolled'})
        
        embeddings_list = []
        
        for idx, image_data in enumerate(images):
            # Decode base64 image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = face_app.get(frame)
            
            if len(faces) == 0:
                return jsonify({'success': False, 'error': f'No face detected in image {idx+1}'})
            
            if len(faces) > 1:
                return jsonify({'success': False, 'error': f'Multiple faces detected in image {idx+1}'})
            
            # Extract embedding
            embedding = faces[0].embedding
            
            if not validate_embedding(embedding):
                return jsonify({'success': False, 'error': f'Invalid embedding in image {idx+1}'})
            
            embeddings_list.append(embedding)
        
        # Average embeddings
        avg_embedding = average_embeddings(embeddings_list)
        
        # Store embedding
        embeddings_db[student_id] = avg_embedding
        save_embeddings(embeddings_db, EMBEDDINGS_FILE)
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} enrolled successfully with {len(images)} samples!'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API endpoint for recognizing a face."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        threshold = data.get('threshold', DEFAULT_THRESHOLD)
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Missing image'})
        
        # Reload database to get latest enrollments
        load_db()
        
        if not embeddings_db:
            return jsonify({'success': False, 'error': 'No enrolled students found'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(frame, max_num=10)
        
        if len(faces) == 0:
            return jsonify({
                'success': True,
                'status': 'NO_FACE',
                'message': 'No face detected'
            })
        
        if len(faces) > 1:
            return jsonify({
                'success': True,
                'status': 'MULTIPLE_FACES',
                'message': f'Multiple faces detected ({len(faces)} faces)',
                'count': len(faces)
            })
        
        # Single face - recognize
        face = faces[0]
        embedding = face.embedding
        
        if not validate_embedding(embedding):
            return jsonify({'success': False, 'error': 'Invalid embedding extracted'})
        
        # Find best match
        student_id, similarity = find_best_match(embedding, embeddings_db, threshold)
        
        if student_id:
            return jsonify({
                'success': True,
                'status': 'VERIFIED',
                'student_id': student_id,
                'similarity': float(similarity),
                'message': f'Verified: {student_id}'
            })
        else:
            return jsonify({
                'success': True,
                'status': 'UNKNOWN',
                'similarity': float(similarity),
                'message': 'Unknown person'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/students', methods=['GET'])
def api_students():
    """Get list of enrolled students."""
    load_db()
    return jsonify({
        'success': True,
        'students': list(embeddings_db.keys()),
        'count': len(embeddings_db)
    })


@app.route('/api/delete_student/<student_id>', methods=['DELETE'])
def api_delete_student(student_id):
    """Delete an enrolled student."""
    try:
        load_db()
        
        if student_id not in embeddings_db:
            return jsonify({'success': False, 'error': 'Student not found'})
        
        del embeddings_db[student_id]
        save_embeddings(embeddings_db, EMBEDDINGS_FILE)
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} deleted successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model(device='cpu')  # Change to 'cuda' for GPU
    load_db()
    
    print("\n" + "="*60)
    print("Face Recognition Web Application")
    print("="*60)
    print(f"Enrolled students: {len(embeddings_db)}")
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
