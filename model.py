"""
Main model.
"""

import tensorflow as tf
from hyperparams import Hyperparams
from layers import encoding, bidirectional_rnn
from loader import tokenize, convert_to_ids
from util import load_glove, embedding_matrix


class Model(object):
    """Main model

    Attributes:
        load_pretrained (bool):         if True, load the model at instantiation
        char_embeddings (tf.Variable):  mxn matrix that contains character embedding
                                        each row represents character token, each column represents embedded dimension
        word_embeddings (tf.Variable):  mxn matrix that contains word embedding
                                        each row represents one word, each column represents embedded dimension
        q_input_char (tf.placeholder):  placeholder for a question converted to character ids
        q_input_word (tf.placeholder):  placeholder for a question converted to word ids
        q_encoded_char (tensor):        a tensor after encoding the input characters to GloVe embedding
        q_encoded_word (tensor):        a tensor after encoding the input words to GloVe embedding
    """
    def __init__(self, load_pretrained=True):

        char_glove = load_glove(Hyperparams.glove_char) if load_pretrained else {}
        word_glove = load_glove(Hyperparams.glove_word) if load_pretrained else {}
        char_embeddings = embedding_matrix(char_glove, 'character')
        word_embeddings = embedding_matrix(word_glove, 'word')

        # create tensor for word & character embeddings
        self.char_embeddings = tf.Variable(tf.constant(char_embeddings), trainable=True, name='char_embeddings')
        self.word_embeddings = tf.Variable(tf.constant(word_embeddings), trainable=False, name='word_embeddings')

        # input placeholders
        self.q_input_char = tf.placeholder(tf.int32, [None, Hyperparams.max_question_c], 'question_c')
        self.q_input_word = tf.placeholder(tf.int32, [None, Hyperparams.max_question_w], 'question_w')

        self.c_input_char = tf.placeholder(tf.int32, [None, Hyperparams.max_context_c], 'context_c')
        self.c_input_word = tf.placeholder(tf.int32, [None, Hyperparams.max_context_w], 'context_w')

        # encode the words using glove
        self.q_encoded_word, q_encoded_char = \
            encoding(self.q_input_word, self.q_input_char, self.word_embeddings, self.char_embeddings)

        # encode the words using glove
        self.c_encoded_word, c_encoded_char = \
            encoding(self.c_input_word, self.c_input_char, self.word_embeddings, self.char_embeddings)

        # store the final layer of bidirectional GRU as character embedding
        self.q_encoded_char = bidirectional_rnn(q_encoded_char, [Hyperparams.max_question_c],
                                                Hyperparams.rnn1_cell_type,  Hyperparams.rnn1_num_units,
                                                Hyperparams.rnn1_num_layers, Hyperparams.rnn1_dropout,
                                                scope = 'Q_char_embed')

        # store the final layer of bidirectional GRU as character embedding
        self.c_encoded_char = bidirectional_rnn(c_encoded_char, [Hyperparams.max_context_c],
                                                Hyperparams.rnn1_cell_type,  Hyperparams.rnn1_num_units,
                                                Hyperparams.rnn1_num_layers, Hyperparams.rnn1_dropout,
                                                scope = 'C_char_embed')


if __name__ == '__main__':
    question = 'What is in front of the Notre Dame Main Building?'
    context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s \
    gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main \
    Building and facing it, is a copper statue of Christ with arms upraised with the \
    legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart.'

    sample_qc = convert_to_ids(question, ttype = 'question', mode='character')
    sample_qw = convert_to_ids(question, ttype = 'question', mode='word')

    sample_cc = convert_to_ids(context, ttype = 'context', mode='character')
    sample_cw = convert_to_ids(context, ttype = 'context', mode='word')

    QuACC = Model(load_pretrained=True)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {
            QuACC.q_input_char: sample_qc.reshape(1, -1),
            QuACC.q_input_word: sample_qw.reshape(1, -1),
            QuACC.c_input_char: sample_cc.reshape(1, -1),
            QuACC.c_input_word: sample_cw.reshape(1, -1),
        }
        index = sess.run([QuACC.q_encoded_char, QuACC.c_encoded_char], feed_dict=feed_dict)

        print(index[0])  # 1 x 80 x (2 x char len)
        print(index[1])  # 1 x 80 x (2 x char len)
