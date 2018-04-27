"""
Main model.
"""

import tensorflow as tf
from hyperparams import Hyperparams as Hp
from layers import bidirectional_rnn, attention_alignment, pointer_net
from loader import convert_to_ids
from util import glove_dict, embedding_matrix
import os
import numpy as np


class Model(object):
    """Main model

    Attributes:
        load_glove (bool): if True, load the model at instantiation
    """
    def __init__(self, batch_size=None, learning_rate=None, load_glove=True, is_training=True):

        # if batch size is not specified, default to value in hyperparams.py
        self.batch_size = batch_size or Hp.batch_size
        self.learning_rate = learning_rate or Hp.learning_rate

        # TODO: implement handling of character embedding
        # load pre-trained GloVe dictionary; create embedding matrix
        if 'word_matrix.npy' not in os.listdir('data'):
            word_glove = glove_dict(Hp.glove_word) if load_glove else {}
            word_matrix = embedding_matrix(word_glove, 'word')
        else:
            word_matrix = np.load('./data/word_matrix.npy')

        # input placeholders (integer encoded sentences) & labels
        with tf.variable_scope('inputs'):
            self.p_word_inputs = tf.placeholder(tf.int32, [self.batch_size, Hp.max_p_words], 'p_words')
            self.q_word_inputs = tf.placeholder(tf.int32, [self.batch_size, Hp.max_q_words], 'q_words')
            self.labels = tf.placeholder(tf.int32, [self.batch_size, 2], 'labels')

        # input length placeholders (actual non-padded length of each sequence in batch; dictates length of unrolling)
        with tf.variable_scope('seq_lengths'):
            self.p_word_lengths = tf.placeholder(tf.int32, [self.batch_size], 'p_words')
            self.q_word_lengths = tf.placeholder(tf.int32, [self.batch_size], 'q_words')

        # create tensor for word embedding matrix, lookup GloVe embeddings of inputs
        with tf.variable_scope('initial_embeddings'):
            self.word_matrix = tf.Variable(tf.constant(word_matrix, dtype=tf.float32), trainable=False, name='word_matrix')
            self.p_word_embeds = tf.nn.embedding_lookup(self.word_matrix, self.p_word_inputs, name='p_word_embeds')
            self.q_word_embeds = tf.nn.embedding_lookup(self.word_matrix, self.q_word_inputs, name='q_word_embeds')

        # encode both paragraph & question using bi-directional RNN
        with tf.variable_scope('p_encodings'):
            self.p_encodings, states = bidirectional_rnn(self.p_word_embeds, self.p_word_lengths, Hp.rnn1_cell,
                                                         Hp.rnn1_layers, Hp.rnn1_units, Hp.rnn1_dropout, is_training)
        with tf.variable_scope('q_encodings'):
            self.q_encodings, _ = bidirectional_rnn(self.q_word_embeds, self.q_word_lengths, Hp.rnn1_cell,
                                                    Hp.rnn1_layers, Hp.rnn1_units, Hp.rnn1_dropout, is_training)

        # proofread questions by attending over itself
        with tf.variable_scope('q_proofread'):
            self.q_pr_out, _, self.q_pr_attn = attention_alignment(self.q_encodings, self.q_word_lengths,
                                                                   self.q_encodings, self.q_word_lengths,
                                                                   Hp.attn_layers, Hp.attn_units,
                                                                   Hp.attn_dropout, Hp.attn_cell,
                                                                   Hp.attn_mech, is_training)
        # create question-aware paragraph encoding using bi-directional RNN with attention
        with tf.variable_scope('q_aware_encoding'):
            self.pq_encoding, _, self.p2q_attn = attention_alignment(self.p_encodings, self.p_word_lengths,
                                                                     self.q_pr_out, self.q_word_lengths,
                                                                     Hp.attn_layers, Hp.attn_units,
                                                                     Hp.attn_dropout, Hp.attn_cell,
                                                                     Hp.attn_mech, is_training)

        # create paragraph encoding with self-matching attention
        # TODO: if decoder is uni-directional, which hidden state from BiRNN should be fed to initial state?
        with tf.variable_scope('self_matching'):
            self.pp_encoding, _, self.p2p_attn = attention_alignment(self.pq_encoding, self.p_word_lengths,
                                                                     self.pq_encoding, self.p_word_lengths,
                                                                     Hp.attn_layers, Hp.attn_units,
                                                                     Hp.attn_dropout, Hp.attn_cell,
                                                                     Hp.attn_mech, is_training)

        # find pointers (in paragraph) to beginning and end of answer to question
        with tf.variable_scope('pointer_net'):
            self.pointer_prob = pointer_net(self.pp_encoding, self.p_word_lengths, 2, self.word_matrix,
                                            Hp.ptr_cell, Hp.ptr_layers, Hp.ptr_units, Hp.ptr_dropout, is_training)
            self.pointers = tf.unstack(tf.argmax(self.pointer_prob, axis=2, output_type=tf.int32))

        # compute loss function
        with tf.variable_scope('loss'):
            loss = tf.zeros(())
            pointers = tf.unstack(self.pointer_prob)
            labels = tf.unstack(self.labels, axis=1)
            equal = []

            for i in range(2):
                loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[i], logits=pointers[i])
                equal.append(tf.equal(self.pointers[i], labels[i]))
            self.loss = tf.reduce_mean(loss)
            self.correct = tf.cast(tf.stack(equal), tf.float32)
            self.all_correct = tf.cast(tf.equal(tf.reduce_sum(self.correct, axis=0), 2), tf.float32)
            self.exact_match = tf.reduce_mean(self.all_correct)

            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


if __name__ == '__main__':

    question = 'What is in front of the Notre Dame Main Building?'
    paragraph = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a ' \
                'golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a ' \
                'copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main ' \
                'Building is the Basilica of the Sacred Heart.'

    sample_pw = convert_to_ids(paragraph, ttype='paragraph', mode='word')
    sample_pw_l = len(sample_pw)
    sample_qw = convert_to_ids(question, ttype='question', mode='word')
    sample_qw_l = len(sample_qw)
    pointers = [37, 42]

    # sample_pc = convert_to_ids(paragraph, ttype='paragraph', mode='character')
    # sample_qc = convert_to_ids(question, ttype='question', mode='character')

    QuACC = Model(batch_size=1, load_glove=True, is_training=False)

    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #     feed_dict = {
    #         QuACC.p_word_inputs: sample_pw.reshape(1, -1),
    #         QuACC.q_word_inputs: sample_qw.reshape(1, -1),
    #         QuACC.p_word_lengths: sample_pw_l.reshape(1, -1),
    #         QuACC.q_word_lengths: sample_qw_l.reshape(1, -1),
    #     }
    #     index = sess.run([QuACC.q_word_embeds, QuACC.p_word_embeds], feed_dict=feed_dict)
    #
    #     print(index[0][1].shape)  # 1 x 80 x (2 x char len)
    #     # print(index[0][1].shape)  # 1 x 80 x (2 x char len)
