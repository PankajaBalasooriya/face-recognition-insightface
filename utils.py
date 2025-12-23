"""
Face Recognition Utilities Module
Provides core functions for face embedding comparison and verification logic.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import os

# Configuration constants
DEFAULT_THRESHOLD = 0.4  # Cosine similarity threshold (lower = stricter)
EMBEDDING_DIM = 512
EMBEDDINGS_FILE = "student_embeddings.pkl"


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector (512-D)
        embedding2: Second embedding vector (512-D)
    
    Returns:
        Cosine similarity score (higher = more similar)
    """
    # Normalize embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Compute cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    return float(similarity)


def find_best_match(
    query_embedding: np.ndarray,
    stored_embeddings: Dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD
) -> Tuple[Optional[str], float]:
    """
    Find the best matching student for a query embedding.
    
    Args:
        query_embedding: Face embedding to match
        stored_embeddings: Dictionary of {student_id: embedding}
        threshold: Minimum similarity threshold for verification
    
    Returns:
        Tuple of (student_id, similarity_score) or (None, best_score)
    """
    if not stored_embeddings:
        return None, 0.0
    
    best_match = None
    best_similarity = -1.0
    
    for student_id, stored_embedding in stored_embeddings.items():
        similarity = cosine_similarity(query_embedding, stored_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = student_id
    
    # Return match only if above threshold
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return None, best_similarity


def save_embeddings(embeddings: Dict[str, np.ndarray], filepath: str = EMBEDDINGS_FILE) -> None:
    """
    Save embeddings dictionary to disk.
    
    Args:
        embeddings: Dictionary of {student_id: embedding}
        filepath: Path to save the embeddings file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {filepath}")


def load_embeddings(filepath: str = EMBEDDINGS_FILE) -> Dict[str, np.ndarray]:
    """
    Load embeddings dictionary from disk.
    
    Args:
        filepath: Path to the embeddings file
    
    Returns:
        Dictionary of {student_id: embedding}
    """
    if not os.path.exists(filepath):
        print(f"No embeddings file found at {filepath}")
        return {}
    
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Loaded {len(embeddings)} student embeddings from {filepath}")
    return embeddings


def average_embeddings(embeddings_list: list) -> np.ndarray:
    """
    Average multiple face embeddings to create a robust representation.
    
    Args:
        embeddings_list: List of embedding arrays
    
    Returns:
        Averaged embedding array
    """
    if not embeddings_list:
        raise ValueError("Cannot average empty list of embeddings")
    
    embeddings_array = np.array(embeddings_list)
    avg_embedding = np.mean(embeddings_array, axis=0)
    
    # Normalize the averaged embedding
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    return avg_embedding


def validate_embedding(embedding: np.ndarray) -> bool:
    """
    Validate that an embedding has the correct shape and properties.
    
    Args:
        embedding: Embedding array to validate
    
    Returns:
        True if valid, False otherwise
    """
    if embedding is None:
        return False
    
    if not isinstance(embedding, np.ndarray):
        return False
    
    if embedding.shape != (EMBEDDING_DIM,):
        return False
    
    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return False
    
    return True