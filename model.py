"""
Main model.
"""

import tensorflow as tf
from hyperparams import Hyperparams as Hp
from layers import bidirectional_rnn
from loader import convert_to_ids
from util import glove_dict, embedding_matrix


class Model:
    """Main model

    Attributes:
        load_glove (bool): if True, load the model at instantiation
    """
    def __init__(self, batch_size=None, load_glove=True, is_training=True):

        # if batch size is not specified, default to value in hyperparams.py
        self.batch_size = batch_size or Hp.batch_size

        # TODO: implement handling of character embedding
        # load pre-trained GloVe dictionary; create embedding matrix
        word_glove = glove_dict(Hp.glove_word) if load_glove else {}
        word_matrix = embedding_matrix(word_glove, 'word')

        # input placeholders (integer encoded sentences)
        with tf.variable_scope('inputs'):
            self.p_word_inputs = tf.placeholder(tf.int32, [self.batch_size, Hp.max_p_words], 'p_words')
            self.q_word_inputs = tf.placeholder(tf.int32, [self.batch_size, Hp.max_q_words], 'q_words')

        # input length placeholders (actual non-padded length of each sequence in batch; dictates length of unrolling)
        with tf.variable_scope('seq_lengths'):
            self.p_word_lengths = tf.placeholder(tf.int32, [self.batch_size], 'p_words')
            self.q_word_lengths = tf.placeholder(tf.int32, [self.batch_size], 'q_words')

        # create tensor for word embedding matrix, lookup GloVe embeddings of inputs
        with tf.variable_scope('initial_embeddings'):
            self.word_matrix = tf.Variable(tf.constant(word_matrix), trainable=False, name='word_matrix')
            self.p_word_embeds = tf.nn.embedding_lookup(self.word_matrix, self.p_word_inputs, name='p_word_embeds')
            self.q_word_embeds = tf.nn.embedding_lookup(self.word_matrix, self.q_word_inputs, name='q_word_embeds')

        # encode both paragraph & question using bi-directional RNN
        with tf.variable_scope('p_encodings'):
            self.p_encodings, states = bidirectional_rnn(self.p_word_embeds, self.p_word_lengths, Hp.rnn1_cell,
                                                         Hp.rnn1_units, Hp.rnn1_layers, Hp.rnn1_dropout, is_training)
        with tf.variable_scope('q_encodings'):
            self.q_encodings, _ = bidirectional_rnn(self.q_word_embeds, self.q_word_lengths, Hp.rnn1_cell,
                                                    Hp.rnn1_units, Hp.rnn1_layers, Hp.rnn1_dropout, is_training)

        # create question-aware paragraph encoding using bi-directional RNN with attention
        with tf.variable_scope('q_aware_encoding'):
            pass

        # create paragraph encoding with self-matching attention
        with tf.variable_scope('self_matching'):
            pass

        # find pointers (in paragraph) to beginning and end of answer to question
        with tf.variable_scope('pointer_net'):
            self.pointers = None

        # loss functions & optimization:
        with tf.variable_scope('loss'):
            self.labels = tf.placeholder(tf.int32, [self.batch_size, 2], 'labels')
            # TODO: implement negative log-likelihood of true label given predicted distribution
            self.neg_log_likelihood = None
            # TODO: implement selection of optimizer, gradient clipping (?)
            self.train_step = tf.train.AdamOptimizer(Hp.learning_rate).minimize(self.neg_log_likelihood)

        # compute accuracy metrics
        with tf.variable_scope('metrics'):
            # TODO: for any P/Q pair, there may be multiple correct answers
            # exact match score (percentage of P/Q pairs where both start & end pointers match)
            match_any = tf.equal(tf.argmax(self.pointers, axis=1, output_type=tf.int32), self.labels)
            match_all = tf.cast(tf.equal(tf.reduce_sum(tf.cast(match_any, tf.int32), axis=1), 2), tf.float64)
            self.exact_match = tf.reduce_mean(match_all)
            # TODO: implement F1 score


if __name__ == '__main__':

    question = 'What is in front of the Notre Dame Main Building?'
    paragraph = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s \
    gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main \
    Building and facing it, is a copper statue of Christ with arms upraised with the \
    legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart.'

    sample_pw = convert_to_ids(paragraph, ttype='paragraph', mode='word')
    sample_qw = convert_to_ids(question, ttype='question', mode='word')

    # sample_pc = convert_to_ids(paragraph, ttype='paragraph', mode='character')
    # sample_qc = convert_to_ids(question, ttype='question', mode='character')


    QuACC = Model(batch_size=1, load_glove=True, is_training=False)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {
            QuACC.p_word_inputs: sample_pw.reshape(1, -1),
            QuACC.q_word_inputs: sample_qw.reshape(1, -1)
        }
        index = sess.run([QuACC.q_word_embeds, QuACC.p_word_embeds], feed_dict=feed_dict)

        print(index[0][1].shape)  # 1 x 80 x (2 x char len)
        # print(index[0][1].shape)  # 1 x 80 x (2 x char len)
