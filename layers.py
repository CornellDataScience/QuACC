"""
Layers of neural network architecture.
"""

import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauAttention, BasicDecoder, GreedyEmbeddingHelper, TrainingHelper
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.rnn import GRUCell

END_TOKEN = 0
START_TOKEN = 1


def bidirectional_rnn(inputs, input_lengths, cell_type, n_layers, n_units, dropout_prob, is_training=True):
    """Bidirectional RNN.

    Args:
        inputs (tensor):        3-dimensional input tensor of shape [batch_size, max_document_length, embedding_size]
        input_lengths (tensor): 1-dimensional tensor of length batch_size specifying actual length of each document
        cell_type (method):     type of RNN cell (e.g. tf.contrib.rnn.GRUCell)
        n_layers (int):         number of layers in the RNN
        n_units (int):          number of units in the RNN cell
        dropout_prob (float):   probability of dropping a node during dropout
        is_training (bool):     whether the model is training or testing
    Returns:
        tensor:                 output tensor or hidden state tensor depending on setting of output
    """
    input_dims = inputs.get_shape().as_list()
    assert len(input_dims) == 3, "Input tensor must be 3-dimensional."

    # instantiate RNN cell; only use dropout during training
    def _rnn_cell():
        keep_prob = 1 - dropout_prob if is_training else 1
        return DropoutWrapper(cell_type(n_units), output_keep_prob=keep_prob)

    # if there is more than one hidden layer, use MultiRNNCell
    if n_layers > 1:
        cell_fw = MultiRNNCell([_rnn_cell() for _ in range(n_layers)])
        cell_bw = MultiRNNCell([_rnn_cell() for _ in range(n_layers)])
    else:
        cell_fw, cell_bw = _rnn_cell(), _rnn_cell()

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, input_lengths, dtype=tf.float32)
    return tf.concat(outputs, axis=2), states


def attention_alignment(inputs, input_lengths, memory, memory_lengths, n_layers, n_units,
                        dropout_prob, cell_type=GRUCell, attention_mechanism=BahdanauAttention, is_training=True):
    """Performs alignment over inputs, atteding memory

    Args:
        inputs (tensor):              Input sequence, with the shape of [Batch x seq_length x dimension]
        input_lengths (tensor):       The length of input sequences. Used for dynamic unrolling
        memory (tensor):              Sequence to attend
        memory_lengths (tensor):      The length of memory. Used for dynamic unrolling
        n_layers (int):               Number of layers in RNN
        n_units  (int):               Number of units in RNN
        dropout_prob (float):         Drop out rate for RNN cell
        cell_type (method):           Type of RNN cell, GRU by default
        attention_mechanism (method): Type of attention mechanism, Bahdanau by default
        is_training (bool):           Whether the model is training or testing

    returns:
        (tensor, tensor, tensor):
    """
    # get the tensor dimension
    batch_size, seq_length, _ = inputs.get_shape().as_list()
    # create a attention over the memory
    attention = attention_mechanism(n_units, memory, memory_sequence_length=memory_lengths, dtype=tf.float32)
    # build an encoder RNN over the input sequence
    if n_layers > 1:
        dropout_prob = 0 if not is_training else dropout_prob
        attention_cell = MultiRNNCell([DropoutWrapper(cell_type(n_units), output_keep_prob=1.0-dropout_prob)
                                       for _ in range(n_layers)])
    else:
        dropout_prob = 0 if not is_training else dropout_prob
        attention_cell = cell_type(n_units)
        attention_cell = DropoutWrapper(attention_cell, output_keep_prob=1.0-dropout_prob)
    # for each input to the next RNN cell, wire the attention mechanism
    a_cell = AttentionWrapper(attention_cell, attention, alignment_history=True)
    # define the initial state
    # TODO: Do we ever feed an init state?
    attention_state = a_cell.zero_state(batch_size, dtype=tf.float32)
    # read input while attending over memory
    helper = TrainingHelper(inputs=inputs, sequence_length=input_lengths)
    decoder = BasicDecoder(a_cell, helper, attention_state)
    # output of the decoder is a new representation of input sentence with attention over the question
    outputs, states, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=seq_length)
    # attention matrix for visualizing heatmap
    aligned = tf.transpose(states.alignment_history.stack(), [1, 0, 2])
    return outputs, states, aligned


def pointer_net(inputs, input_lengths, n_pointers, word_matrix, cell_type, n_layers, n_units,
                dropout_prob, is_training=True):
    """Pointer network.

    Args:
        inputs (tensor):        Inputs to pointer network (typically output of previous RNN)
        input_lengths (tensor): Actual non-padded lengths of each input sequence
        n_pointers (int):       Number of pointers to generate
        word_matrix (tensor):   Embedding matrix of word vectors
        cell_type (method):     Cell type to use
        n_layers (int):         Number of layers in RNN (same for encoder & decoder)
        n_units (int):          Number of units in RNN cell (same for encoder & decoder)
        dropout_prob (float):   Dropout probability
        is_training (bool):     Whether the model is training or testing
    """
    batch_size, seq_length, _ = inputs.get_shape().as_list()
    vocab_size = word_matrix.get_shape().as_list()[0]

    # instantiate RNN cell; only use dropout during training
    def _rnn_cell():
        keep_prob = 1 - dropout_prob if is_training else 1
        return DropoutWrapper(cell_type(n_units), output_keep_prob=keep_prob)

    enc_cell = MultiRNNCell([_rnn_cell() for _ in range(n_layers)]) if n_layers > 1 else _rnn_cell()
    encoded, _ = tf.nn.dynamic_rnn(enc_cell, inputs, input_lengths, dtype=tf.float32)

    attention = BahdanauAttention(n_units, encoded, memory_sequence_length=input_lengths)
    # TODO: find permanent solution (InferenceHelper?)
    start_tokens = tf.constant(START_TOKEN, shape=[batch_size], dtype=tf.int32)
    helper = GreedyEmbeddingHelper(word_matrix, start_tokens, END_TOKEN)

    dec_cell = MultiRNNCell([_rnn_cell() for _ in range(n_layers)]) if n_layers > 1 else _rnn_cell()
    attn_cell = AttentionWrapper(dec_cell, attention, alignment_history=True)
    out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size)
    decoder = BasicDecoder(out_cell, helper, attn_cell.zero_state(batch_size, tf.float32))
    _, states, _ = dynamic_decode(decoder, maximum_iterations=n_pointers, impute_finished=True)
    probs = tf.reshape(states.alignment_history.stack(), [n_pointers, batch_size, seq_length])
    return probs
