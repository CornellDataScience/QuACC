"""
Layers of neural network architecture.
"""

import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell


def embedding(word, char, word_embeddings, char_embeddings, scope='embedding'):
    """Encode the list of word ids and character ids to pretrained embeddings using tf.nn.embedding_lookup
    Args:
        word (list):                   list of word ids
        char (list):                   list of char ids
        word_embeddings (tf.Variable): pretrained [(size of vocabulary) x (embedding dimension; default 300)] matrix
        char_embeddings (tf.Variable): pretrained [(size of vocabulary) x (embedding dimension; default 300)] matrix
        scope (str):                   tensorflow variable scope

    Returns:
        word_encoding (tensor)
        char_encoding (tensor)
    """
    with tf.variable_scope(scope):
        word_embeds = tf.nn.embedding_lookup(word_embeddings, word)
        char_embeds = tf.nn.embedding_lookup(char_embeddings, char)
        return word_embeds, char_embeds


def bidirectional_rnn(inputs, input_lengths, cell_type, num_units, num_layers, dropout_prob, is_training=True):
    """Bidirectional RNN.

    Args:
        inputs (tensor):        3-dimensional input tensor of shape [batch_size, max_document_length, embedding_size]
        input_lengths (tensor): 1-dimensional tensor of length batch_size specifying actual length of each document
        cell_type (method):     type of RNN cell (e.g. tf.contrib.rnn.GRUCell)
        num_units (int):        number of units in the RNN cell
        num_layers (int):       number of layers in the RNN
        dropout_prob (float):   probability of dropping a node during dropout
        is_training (bool):     whether the model is training or testing

    Returns:
        tensor:                 output tensor or hidden state tensor depending on setting of output
    """
    input_dims = inputs.get_shape().as_list()
    assert len(input_dims) == 3, "Input tensor must be 3-dimensional."

    def rnn_cell():
        # only use dropout during training
        keep_prob = 1 - dropout_prob if is_training else 1
        return DropoutWrapper(cell_type(num_units), output_keep_prob=keep_prob)

    if num_layers > 1:
        # if there is more than one hidden layer, use MultiRNNCell
        cell_fw = MultiRNNCell([rnn_cell() for _ in range(num_layers)])
        cell_bw = MultiRNNCell([rnn_cell() for _ in range(num_layers)])
    else:
        cell_fw, cell_bw = rnn_cell(), rnn_cell()

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, input_lengths, dtype=tf.float64)
    return tf.concat(outputs, axis=2), states


def simple_attention(inputs, scope='attention'):
    pass
