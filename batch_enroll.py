"""
Batch Student Enrollment from Photos
Enroll multiple students from a folder of portrait photos.
"""

import os
import cv2
from enrollment import StudentEnrollment


def batch_enroll_from_folder(folder_path: str, device: str = 'cpu'):
    """
    Enroll all students from a folder of photos.
    
    Photo naming convention:
    - StudentID.jpg (e.g., STU001.jpg, JOHN_DOE.png)
    - The filename (without extension) becomes the student ID
    
    Args:
        folder_path: Path to folder containing portrait photos
        device: 'cpu' or 'cuda'
    """
    print("\n" + "="*60)
    print("BATCH ENROLLMENT FROM PHOTOS")
    print("="*60)
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return
    
    # Get all image files
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in supported_formats
    ]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        print(f"  Supported formats: {', '.join(supported_formats)}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Folder: {folder_path}\n")
    
    # Initialize enrollment system
    enroller = StudentEnrollment(device=device)
    
    # Statistics
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each image
    for idx, filename in enumerate(image_files, 1):
        # Extract student ID from filename
        student_id = os.path.splitext(filename)[0]
        photo_path = os.path.join(folder_path, filename)
        
        print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
        print(f"Student ID: {student_id}")
        
        # Check if already enrolled
        if student_id in enroller.embeddings_db:
            print(f"Student {student_id} already enrolled - skipping")
            skipped += 1
            continue
        
        # Enroll student
        success = enroller.enroll_from_photo(student_id, photo_path)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH ENROLLMENT SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successfully enrolled: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already enrolled): {skipped}")
    print("="*60)
    
    if successful > 0:
        print(f"\n{successful} students enrolled successfully!")
        print(f"Database saved to: {enroller.embeddings_db}")


def main():
    """Main batch enrollment interface."""
    print("\n" + "="*60)
    print("BATCH ENROLLMENT FROM PORTRAIT PHOTOS")
    print("="*60)
    print("\nThis tool enrolls multiple students from a folder of photos.")
    print("\nPhoto naming convention:")
    print("  - StudentID.jpg (e.g., STU001.jpg, ALICE_SMITH.png)")
    print("  - The filename (without extension) becomes the student ID")
    print("\nPhoto requirements:")
    print("  - Clear frontal face view")
    print("  - Good lighting")
    print("  - Only one person per photo")
    print("  - Recommended: passport-style photos")
    
    # Get folder path
    print("\n" + "="*60)
    folder_path = input("\nEnter path to photos folder (or drag & drop): ").strip()
    folder_path = folder_path.strip('"').strip("'")
    
    # Check for GPU
    device = 'cpu'
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available_providers:
            use_gpu = input("\nGPU detected. Use GPU acceleration? (y/n): ").strip().lower()
            if use_gpu == 'y':
                device = 'cuda'
                print("GPU acceleration enabled")
    except ImportError:
        pass
    
    # Confirm before proceeding
    print("\n" + "="*60)
    confirm = input("Start batch enrollment? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Batch enrollment cancelled.")
        return
    
    # Run batch enrollment
    batch_enroll_from_folder(folder_path, device)


if __name__ == "__main__":
    main()