"""
Layers
"""

import tensorflow as tf
import numpy as np


def encoding(word, char, word_embeddings, char_embeddings, scope="embedding"):
    """Encode the list of word ids and character ids to pretrained embeddings using tf.nn.embedding_lookup
    Args:
        word (list):                  list of word ids
        char (list):                  list of char ids
        word_embeddings (np.ndarray): pretrained nxm matrix where n = number of vocabulary and
                                      m = embedding dimension (default 300)
        char_embeddings (np.ndarray): pretrained nxm matrix where n = number of vocabulary and
                                      m = embedding dimension (default 300)
        scope (str):                  tensorflow variable scope
    
    Returns:
        word_encoding (tensor)
        char_encoding (tensor)
    """
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding


def bidirectional_rnn(input):
    pass


def cnn_embedding(input):
    pass
