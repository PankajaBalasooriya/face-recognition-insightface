"""
Live Face Recognition Module
Performs continuous face verification during exam proctoring.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
from typing import Tuple, List
from utils import (
    load_embeddings,
    find_best_match,
    validate_embedding,
    DEFAULT_THRESHOLD,
    EMBEDDINGS_FILE
)


class FaceRecognitionProctor:
    """Real-time face recognition for exam proctoring."""
    
    # Status constants
    STATUS_VERIFIED = "VERIFIED"
    STATUS_UNKNOWN = "UNKNOWN PERSON"
    STATUS_MULTIPLE_FACES = "MULTIPLE FACES DETECTED"
    STATUS_NO_FACE = "NO FACE DETECTED"
    
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, device: str = 'cpu'):
        """
        Initialize face recognition system.
        
        Args:
            threshold: Similarity threshold for verification (lower = stricter)
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.threshold = threshold
        
        print("Initializing InsightFace model...")
        
        # Configure providers based on device
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
            print("🚀 Using GPU acceleration (CUDA)")
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print("⚙️  Using CPU processing")
        
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("✓ Model loaded successfully")
        
        # Load enrolled students
        self.embeddings_db = load_embeddings(EMBEDDINGS_FILE)
        
        if not self.embeddings_db:
            print("\n⚠ WARNING: No enrolled students found!")
            print("Please run enrollment.py first to enroll students.\n")
        
        # Performance metrics
        self.fps_list = []
        self.last_time = time.time()
    
    def verify_frame(self, frame: np.ndarray) -> Tuple[str, List[dict]]:
        """
        Verify all faces in a single frame.
        
        Args:
            frame: Input BGR image from camera
        
        Returns:
            Tuple of (status, face_data_list)
            face_data_list contains dicts with bbox, student_id, similarity
        """
        faces = self.app.get(frame)
        face_data_list = []
        
        if len(faces) == 0:
            return self.STATUS_NO_FACE, face_data_list
        
        if len(faces) > 1:
            # Multiple faces detected - create data for each
            for face in faces:
                face_data_list.append({
                    'bbox': face.bbox.astype(int),
                    'student_id': None,
                    'similarity': 0.0
                })
            return self.STATUS_MULTIPLE_FACES, face_data_list
        
        # Single face - verify identity
        face = faces[0]
        embedding = face.normed_embedding
        
        face_data = {
            'bbox': face.bbox.astype(int),
            'student_id': None,
            'similarity': 0.0
        }
        
        if validate_embedding(embedding):
            student_id, similarity = find_best_match(
                embedding, 
                self.embeddings_db, 
                self.threshold
            )
            
            face_data['student_id'] = student_id
            face_data['similarity'] = similarity
            
            if student_id is not None:
                face_data_list.append(face_data)
                return self.STATUS_VERIFIED, face_data_list
            else:
                face_data_list.append(face_data)
                return self.STATUS_UNKNOWN, face_data_list
        
        face_data_list.append(face_data)
        return self.STATUS_UNKNOWN, face_data_list
    
    def draw_results(self, frame: np.ndarray, status: str, face_data_list: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            status: Verification status
            face_data_list: List of face data dictionaries
        
        Returns:
            Annotated frame
        """
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Define colors for different statuses
        color_map = {
            self.STATUS_VERIFIED: (0, 255, 0),      # Green
            self.STATUS_UNKNOWN: (0, 0, 255),        # Red
            self.STATUS_MULTIPLE_FACES: (0, 165, 255),  # Orange
            self.STATUS_NO_FACE: (128, 128, 128)     # Gray
        }
        
        color = color_map.get(status, (255, 255, 255))
        
        # Draw status banner
        banner_height = 80
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Status text
        cv2.putText(display_frame, status, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw face data
        for face_data in face_data_list:
            bbox = face_data['bbox']
            student_id = face_data['student_id']
            similarity = face_data['similarity']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, 3)
            
            # Draw label background
            if student_id:
                label = f"{student_id} ({similarity:.2f})"
                label_color = (0, 255, 0)
            else:
                label = f"UNKNOWN ({similarity:.2f})" if similarity > 0 else "UNKNOWN"
                label_color = (0, 0, 255)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, (bbox[0], bbox[1] - 35), 
                         (bbox[0] + label_size[0] + 10, bbox[1]), label_color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (bbox[0] + 5, bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        if self.fps_list:
            avg_fps = sum(self.fps_list[-30:]) / len(self.fps_list[-30:])
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (width - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(display_frame, "Press 'q' to quit | 'r' to reload database", 
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display_frame
    
    def run(self):
        """Run continuous face recognition."""
        print("\n" + "="*60)
        print("EXAM PROCTORING SYSTEM - ACTIVE")
        print("="*60)
        print(f"Threshold: {self.threshold}")
        print(f"Enrolled students: {len(self.embeddings_db)}")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Performance optimization - reduce frame processing if needed
        frame_skip = 0  # Process every frame for real-time proctoring
        frame_count = 0
        
        print("✓ Camera opened. Starting verification...")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Process frame
            if frame_count % (frame_skip + 1) == 0:
                status, face_data_list = self.verify_frame(frame)
                
                # Log important events
                if status == self.STATUS_MULTIPLE_FACES:
                    print(f"⚠ [{time.strftime('%H:%M:%S')}] ALERT: Multiple faces detected!")
                elif status == self.STATUS_UNKNOWN:
                    print(f"⚠ [{time.strftime('%H:%M:%S')}] ALERT: Unknown person detected!")
                elif status == self.STATUS_VERIFIED and face_data_list:
                    student_id = face_data_list[0]['student_id']
                    if frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                        print(f"✓ [{time.strftime('%H:%M:%S')}] Verified: {student_id}")
            
            # Draw results
            display_frame = self.draw_results(frame, status, face_data_list)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time)
            self.fps_list.append(fps)
            self.last_time = current_time
            
            # Display
            cv2.imshow('Exam Proctoring - Face Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nShutting down proctoring system...")
                break
            elif key == ord('r'):
                print("\nReloading student database...")
                self.embeddings_db = load_embeddings(EMBEDDINGS_FILE)
                print(f"✓ Loaded {len(self.embeddings_db)} students")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        if self.fps_list:
            print(f"Average FPS: {sum(self.fps_list) / len(self.fps_list):.1f}")
        print(f"Total frames processed: {frame_count}")
        print("="*60)


def main():
    """Main recognition interface."""
    print("\n" + "="*60)
    print("EXAM PROCTORING SYSTEM - FACE RECOGNITION")
    print("="*60)
    
    # Configuration
    threshold = DEFAULT_THRESHOLD
    device = 'cpu'
    
    # Allow user to configure threshold
    config_choice = input("\nUse default threshold (0.4)? (y/n): ").strip().lower()
    if config_choice == 'n':
        try:
            threshold = float(input("Enter threshold (0.0-1.0, lower=stricter): ").strip())
            threshold = max(0.0, min(1.0, threshold))
            print(f"✓ Threshold set to {threshold}")
        except ValueError:
            print("✗ Invalid input. Using default threshold.")
            threshold = DEFAULT_THRESHOLD
    
    # Check for GPU availability
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available_providers:
            print("\n✓ GPU (CUDA) detected!")
            use_gpu = input("Use GPU acceleration? (y/n): ").strip().lower()
            if use_gpu == 'y':
                device = 'cuda'
                print("🚀 GPU acceleration enabled")
        else:
            print("\nℹ️  GPU not available. Using CPU.")
            print("   To enable GPU: pip install onnxruntime-gpu")
    except ImportError:
        print("\nℹ️  Using CPU processing")
    
    # Initialize and run
    proctor = FaceRecognitionProctor(threshold=threshold, device=device)
    proctor.run()


if __name__ == "__main__":
    main()