"""
All hyperparameters. Implemented as a class for convenience, but operates identically to an argument parser.
"""

import pickle
import tensorflow as tf


class Hyperparams:

    # pre-trained models
    glove_word = './models/glove.840B.300d.txt'
    glove_char = './models/glove.840B.300d-char.txt'

    # data
    data_dir = './data'

    # training
    dropout = 0.2
    optimizer = 'adam'

    # architecture
    emb_size = 300
    rnn1_cell = tf.contrib.rnn.GRUCell
    rnn1_layers = 3
    rnn1_units = 80
    rnn1_dropout = 0.2

    # SQuAD related info
    with open('./data/word2id-dict.pkl', 'rb') as f:
        word2id = pickle.load(f)
    with open('./data/id2word-dict.pkl', 'rb') as f:
        id2word = pickle.load(f)
    with open('./data/char2id-dict.pkl', 'rb') as f:
        char2id = pickle.load(f)
    with open('./data/id2char-dict.pkl', 'rb') as f:
        id2char = pickle.load(f)

    # vocabulary sizes
    word_vocab_size = len(word2id)
    char_vocab_size = len(char2id)

    # input size
    max_q_chars = 80
    max_q_words = 30

    # input size
    max_p_chars = 1000
    max_p_words = 500
