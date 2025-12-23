"""
Student Enrollment Module
Captures multiple face images, extracts embeddings, and stores averaged embeddings.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
from utils import (
    save_embeddings, 
    load_embeddings, 
    average_embeddings,
    validate_embedding,
    EMBEDDINGS_FILE
)


class StudentEnrollment:
    """Handles student enrollment process for face recognition."""
    
    def __init__(self, num_samples: int = 10, device: str = 'cpu'):
        """
        Initialize enrollment system.
        
        Args:
            num_samples: Number of face samples to capture per student
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.num_samples = num_samples
        
        print("Initializing InsightFace model...")
        
        # Configure providers based on device
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
            print("Using GPU acceleration (CUDA)")
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print("Using CPU processing")
        
        self.app = FaceAnalysis(
            name='buffalo_l',  # High-accuracy model
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("Model loaded successfully")
        
        # Load existing embeddings
        self.embeddings_db = load_embeddings(EMBEDDINGS_FILE)
    
    def capture_face_samples(self, student_id: str) -> bool:
        """
        Capture multiple face samples for a student.
        
        Args:
            student_id: Unique identifier for the student
        
        Returns:
            True if enrollment successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"ENROLLING STUDENT: {student_id}")
        print(f"{'='*60}")
        print(f"Please look at the camera. We will capture {self.num_samples} samples.")
        print("Press 'q' to cancel enrollment\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        embeddings_list = []
        samples_captured = 0
        last_capture_time = 0
        capture_interval = 0.5  # Seconds between captures
        
        while samples_captured < self.num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = self.app.get(frame)
            
            # Display frame
            display_frame = frame.copy()
            current_time = time.time()
            
            if len(faces) == 0:
                cv2.putText(display_frame, "NO FACE DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(display_frame, "MULTIPLE FACES - Please ensure only one person", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), (0, 165, 255), 2)
            else:
                face = faces[0]
                bbox = face.bbox.astype(int)
                
                # Draw bounding box
                color = (0, 255, 0) if samples_captured < self.num_samples else (255, 0, 0)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), color, 2)
                
                # Capture sample if enough time has passed
                if current_time - last_capture_time >= capture_interval:
                    embedding = face.normed_embedding
                    
                    if validate_embedding(embedding):
                        embeddings_list.append(embedding)
                        samples_captured += 1
                        last_capture_time = current_time
                        print(f"Sample {samples_captured}/{self.num_samples} captured")
                
                # Display progress
                progress_text = f"Samples: {samples_captured}/{self.num_samples}"
                cv2.putText(display_frame, progress_text, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(display_frame, "Press 'q' to cancel", (50, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Student Enrollment', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nEnrollment cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Process captured embeddings
        if len(embeddings_list) == self.num_samples:
            avg_embedding = average_embeddings(embeddings_list)
            self.embeddings_db[student_id] = avg_embedding
            save_embeddings(self.embeddings_db, EMBEDDINGS_FILE)
            
            print(f"\nSuccessfully enrolled {student_id}")
            print(f"{self.num_samples} samples averaged into robust embedding")
            return True
        else:
            print(f"\nEnrollment failed: Only captured {len(embeddings_list)} samples")
            return False
    
    def list_enrolled_students(self):
        """Display all enrolled students."""
        if not self.embeddings_db:
            print("\nNo students enrolled yet.")
        else:
            print(f"\n{'='*60}")
            print("ENROLLED STUDENTS:")
            print(f"{'='*60}")
            for idx, student_id in enumerate(self.embeddings_db.keys(), 1):
                print(f"{idx}. {student_id}")
            print(f"{'='*60}")
    
    def remove_student(self, student_id: str) -> bool:
        """
        Remove a student from the database.
        
        Args:
            student_id: Student ID to remove
        
        Returns:
            True if removed, False if not found
        """
        if student_id in self.embeddings_db:
            del self.embeddings_db[student_id]
            save_embeddings(self.embeddings_db, EMBEDDINGS_FILE)
            print(f"Removed {student_id} from database")
            return True
        else:
            print(f"Student {student_id} not found in database")
            return False


def main():
    """Main enrollment interface."""
    print("\n" + "="*60)
    print("EXAM PROCTORING SYSTEM - STUDENT ENROLLMENT")
    print("="*60)
    
    # Initialize enrollment system
    enroller = StudentEnrollment(num_samples=10, device='cpu')
    
    while True:
        print("\n" + "="*60)
        print("MENU:")
        print("1. Enroll new student")
        print("2. List enrolled students")
        print("3. Remove student")
        print("4. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            student_id = input("\nEnter student ID (e.g., STU001): ").strip()
            if not student_id:
                print("Invalid student ID")
                continue
            
            if student_id in enroller.embeddings_db:
                overwrite = input(f"Student {student_id} already enrolled. Overwrite? (y/n): ").strip().lower()
                if overwrite != 'y':
                    continue
            
            enroller.capture_face_samples(student_id)
        
        elif choice == '2':
            enroller.list_enrolled_students()
        
        elif choice == '3':
            student_id = input("\nEnter student ID to remove: ").strip()
            enroller.remove_student(student_id)
        
        elif choice == '4':
            print("\nExiting enrollment system...")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()