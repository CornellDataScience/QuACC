"""
Main model.
"""

import tensorflow as tf
from hyperparams import Hyperparams as Hp
from layers import bidirectional_rnn
from loader import tokenize, convert_to_ids
from util import load_glove, embedding_matrix


class Model:
    """Main model

    Attributes:
        load_pretrained (bool):         if True, load the model at instantiation
    """
    def __init__(self, load_pretrained=True, is_training=True):

        # TODO: implement handling of character embeddings
        # char_glove = load_glove(Hp.glove_char) if load_pretrained else {}
        # char_embeddings = embedding_matrix(char_glove, 'character')
        # self.q_input_char = tf.placeholder(tf.int32, [None, Hp.max_question_c], 'question_c')
        # self.p_input_char = tf.placeholder(tf.int32, [None, Hp.max_paragraph_c], 'paragraph_c')
        # self.char_embeddings = tf.Variable(tf.constant(char_embeddings), trainable=True, name='char_embeddings')

        # load pre-trained GloVe dictionary; create embedding matrix
        word_glove = load_glove(Hp.glove_word) if load_pretrained else {}
        word_matrix = embedding_matrix(word_glove, 'word')

        # input placeholders (integer encoded sentences)
        with tf.variable_scope('inputs'):
            self.p_word_inputs = tf.placeholder(tf.int32, [Hp.batch_size, Hp.max_p_words], 'p_words')
            self.q_word_inputs = tf.placeholder(tf.int32, [Hp.batch_size, Hp.max_q_words], 'q_words')

        # input length placeholders (actual non-padded length of each sequence in batch; dictates length of unrolling)
        with tf.variable_scope('seq_lengths'):
            self.p_word_lengths = tf.placeholder(tf.int32, [Hp.batch_size], 'p_words')
            self.q_word_lengths = tf.placeholder(tf.int32, [Hp.batch_size], 'q_words')

        # create tensor for word embedding matrix, lookup GloVe embeddings of inputs
        with tf.variable_scope('initial_embeddings'):
            self.word_matrix = tf.Variable(tf.constant(word_matrix), trainable=False, name='word_matrix')
            self.q_word_embeds = tf.nn.embedding_lookup(self.word_matrix, self.q_word_inputs, name='q_word_embeds')
            self.p_word_embeds = tf.nn.embedding_lookup(self.word_matrix, self.p_word_inputs, name='p_word_embeds')

        # encode both paragraph & question using bi-directional RNN
        with tf.variable_scope('encodings'):
            self.q_encodings = bidirectional_rnn(self.p_word_embeds, self.p_word_lengths, Hp.rnn1_cell, Hp.rnn1_units,
                                                 Hp.rnn1_layers, Hp.rnn1_dropout, is_training)
            self.p_encodings = bidirectional_rnn(self.p_word_embeds, self.p_word_inputs, Hp.rnn1_cell, Hp.rnn1_units,
                                                 Hp.rnn1_layers, Hp.rnn1_dropout, is_training)


if __name__ == '__main__':

    question = 'What is in front of the Notre Dame Main Building?'
    paragraph = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s \
    gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main \
    Building and facing it, is a copper statue of Christ with arms upraised with the \
    legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart.'

    # sample_qc = convert_to_ids(question, ttype='question', mode='character')
    sample_qw = convert_to_ids(question, ttype='question', mode='word')

    # sample_pc = convert_to_ids(context, ttype='context', mode='character')
    sample_pw = convert_to_ids(paragraph, ttype='paragraph', mode='word')

    QuACC = Model(load_pretrained=True)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {
            QuACC.q_word_inputs: sample_qw.reshape(1, -1),
            QuACC.p_word_inputs: sample_pw.reshape(1, -1)
        }
        index = sess.run([QuACC.q_word_embeds, QuACC.p_word_embeds], feed_dict=feed_dict)

        print(index[0])  # 1 x 80 x (2 x char len)
        print(index[1])  # 1 x 80 x (2 x char len)
