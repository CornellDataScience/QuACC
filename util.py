"""
Miscellaneous helper methods.
"""

import numpy as np
from hyperparams import Hyperparams
from tqdm import tqdm
import argparse
import sys

def glove_dict(directory):
    """Load glove model.

    Args:
        directory (str): the directory to the glove
    Returns:
        dictionary of words (or characters) to the corresponding embedding
    """
    print('Loading embeddings from {}...'.format(directory))
    embedding_vectors = {}
    with open(directory, 'r') as f:
        for line in tqdm(f):
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            char = line_split[0]
            embedding_vectors[char] = vec
    return embedding_vectors


def embedding_matrix(embedding_vectors, mode):
    """Create embedding matrix.

    Args:
        embedding_vectors (dict): dictionary mapping key (either character or word) to embedding vector
        mode (str):               either 'character' or 'word'
    Returns:
        np.ndarray:               embedding matrix
    """
    # TODO: implementation does not match use case
    assert mode in {'character', 'word'}, "Invalid mode."
    if mode == 'character':
        key2id = Hyperparams.char2id
    elif mode == 'word':
        key2id = Hyperparams.word2id

    embedding = np.zeros((len(key2id) + 2, Hyperparams.emb_size))  # extra rows for unknown tokens and starting token
    for key, i in tqdm(key2id.items()):
        if key in embedding_vectors:
            embedding[i] = embedding_vectors.get(key)              # insert embedding vector into matrix
    return embedding

if __name__ == '__main__':
    word_glove = glove_dict(Hyperparams.glove_word)
    word_matrix = embedding_matrix(word_glove, 'word')
    np.save(Hyperparams.data_dir, word_matrix)
