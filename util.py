"""
Miscellaneous helper methods.
"""

import numpy as np


def load_glove(directory):
    """ load glove model
    Args:
        directory (str): the directory to the glove
    Returns:
        dictionary of words (or characters) to the corresponding embedding
    """
    embedding_vectors = {}
    with open(directory, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            char = line_split[0]
            embedding_vectors[char] = vec
    return embedding_vectors
