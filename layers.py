"""
Layers of neural network architecture.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.contrib.seq2seq import TrainingHelper, BasicDecoder, dynamic_decode


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

    # instantiate RNN cell; only use dropout during training
    def rnn_cell():
        keep_prob = 1 - dropout_prob if is_training else 1
        return DropoutWrapper(cell_type(num_units), output_keep_prob=keep_prob)

    # if there is more than one hidden layer, use MultiRNNCell
    if num_layers > 1:
        cell_fw = MultiRNNCell([rnn_cell() for _ in range(num_layers)])
        cell_bw = MultiRNNCell([rnn_cell() for _ in range(num_layers)])
    else:
        cell_fw, cell_bw = rnn_cell(), rnn_cell()

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, input_lengths, dtype=tf.float64)
    return tf.concat(outputs, axis=2), states


def attention(inputs, memory, attn_size):
    """attention layer.

    Args:
        inputs (tf.Tensor):        3-dimensional tensor of shape [batch_size, max_seq_length, dimension]
        memory (tf.Tensor):        3-dimensional tensor of shape [batch_size, max_seq_length, dimension]
        attn_size (int):           hidden size used to compute attention scores
    Returns:
        tf.Tensor:                 attention matrix of shape [batch_size, input_sequence, memory_sequence]
    """
    bi, n, d = inputs.get_shape().as_list()
    b, m, d = memory.get_shape().as_list()
    assert bi == b, "Inputs and memory must have same batch size."

    # trainable variables for attention decoder
    memory_weights = tf.get_variable('wQ', shape=[d, attn_size], dtype=tf.float64)
    input_weights = tf.get_variable('wP', shape=[d, attn_size], dtype=tf.float64)
    attn_weights = tf.get_variable('v', shape=[attn_size], dtype=tf.float64)

    # TODO: Account for varying length inputs? Does it matter if extra entries are zero-padded?
    # compute attention matrix
    weighted_inputs = tf.tensordot(inputs, input_weights, axes=[[2], [0]])
    weighted_memory = tf.tensordot(memory, memory_weights, axes=[[2], [0]])
    tiled_inputs = tf.tile(tf.expand_dims(weighted_inputs, axis=2), [1, 1, m, 1])
    tiled_memory = tf.tile(tf.expand_dims(weighted_memory, axis=1), [1, n, 1, 1])
    attn_matrix = tf.tensordot(tf.tanh(tiled_inputs + tiled_memory), attn_weights, axes=[[3], [0]])
    attn_matrix = tf.nn.softmax(attn_matrix, axis=2)
    assert [b, n, m] == attn_matrix.get_shape().as_list(), "Attention matrix must have shape [batch, n, m]."

    return attn_matrix


def attention_decoder(inputs, memory, input_lengths, initial_state, cell_type, num_units, num_layers, attn_size,
                      dropout_prob, is_training=True):
    """RNN decoder with attention.

    Args:
        inputs (tf.Tensor):        3-dimensional tensor of shape [batch_size, max_seq_length, dimension]
        memory (tf.Tensor):        3-dimensional tensor of shape [batch_size, max_seq_length, dimension]
        input_lengths (tf.Tensor): 1-dimensional tensor of length batch_size specifying actual length of each input
        initial_state (tuple):     tuple of initial states
        cell_type (method):        type of RNN cell (e.g. tf.contrib.rnn.GRUCell)
        num_units (int):           number of units in RNN cell
        num_layers (int):          number of layers in RNN
        attn_size (int):           hidden size used to compute attention scores
        dropout_prob (float):      probability of dropping a node during dropout
        is_training (bool):        whether the model is training or testing
    Returns:
        tf.Tensor:                 output of RNN
        tf.Tensor:                 final hidden state of RNN
    """
    attn_matrix = attention(inputs, memory, attn_size)

    # inputs to RNN are original inputs concatenated with context vector
    # TODO: check correctness of context vector
    context_vector = tf.reduce_sum(tf.tensordot(attn_matrix, memory, axes=[[2], [1]]), axis=2)
    rnn_inputs = tf.concat((inputs, context_vector), axis=2)

    # instantiate RNN cell; only use dropout during training
    def rnn_cell():
        keep_prob = 1 - dropout_prob if is_training else 1
        return DropoutWrapper(cell_type(num_units), output_keep_prob=keep_prob)

    decoder_cell = MultiRNNCell([rnn_cell() for _ in range(num_layers)]) if num_layers > 1 else rnn_cell()
    outputs, states = tf.nn.dynamic_rnn(decoder_cell, rnn_inputs, input_lengths, initial_state)
    return outputs, states


def pointer_net(inputs, labels, initial_state, cell_type, num_units, num_layers, dropout_prob, is_training):
    """Pointer network.

    Args:
    Returns:
    """
    batch = inputs.get_shape().as_list()[0]

    # instantiate RNN cell; only use dropout during training
    def rnn_cell():
        keep_prob = 1 - dropout_prob if is_training else 1
        return DropoutWrapper(cell_type(num_units), output_keep_prob=keep_prob)

    decoder_cell = MultiRNNCell([rnn_cell() for _ in range(num_layers)]) if num_layers > 1 else rnn_cell()
    helper = TrainingHelper(labels, tf.constant(np.full(batch, 2)), time_major=False)
    decoder = BasicDecoder(decoder_cell, helper, initial_state)
    outputs, _ = dynamic_decode(decoder)
    return outputs
