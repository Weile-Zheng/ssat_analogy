import numpy as np


def cosine_similarity(vector1, vector2):
    # Calculate the cosine distance between the two vectors using NumPy
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
