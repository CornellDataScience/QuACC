"""
Layers
"""
import tensorflow as tf


def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
    """Emcode the list of word ids and character ids to pretrained embeddings using
    tf.nn.embedding_lookup
    Args:
        word: list of word ids
        char: list of char ids
        word_embeddings: pretrained nxm matrix where n = number of vocabulary
        and m = embedding dimension (300)
        char_embeddings: pretrained nxm matrix where n = number of vocabulary
        and m = embedding dimension (300)
        scope:

    Returns:
        word_encoding, char_encoding
    """
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding


def bidirectional_rnn(input):
    pass


def cnn_embedding(input):
    pass
