"""
Main model.
"""

from hyperparams import Hyperparams
import tensorflow as tf
from layers import *
from util import load_glove
from loader import *


class Model(object):
    """Main model

    Attributes:
        load_pretrained (bool):         if True, load the model at the instantiation
        char_embeddings (tf.Variable):  mxn matrix that contains character embedding
                                        each row represents character token and
                                        columns represent embedded dimension
        word_embeddings (tf.Variable):  mxn matrix that contains word embedding
                                        each row represents one word and
                                        columns represent embedded dimension
        q_input_char (tf.placeholder):  placeholder for a question converted to
                                        character ids
        q_input_word (tf.placeholder):  placeholder for a question converted to
                                        word ids
        q_encoded_char (tensor):        a tensor after encoding the input character
                                        to the glove embedding
        q_encoded_word (tensor):        a tensor after encoding the input words
                                        to the glove embedding
    """
    def __init__(self, load_pretrained=True):
        if load_pretrained:
            print('start loading character embedding')
            char_embeddings = np.zeros([Hyperparams.char_vocab_size+1, Hyperparams.emb_size])
            char_glove = load_glove(Hyperparams.glove_char)  # load matrixå
            for char, i in Hyperparams.char2id.items():
                if char in char_glove:
                    char_embeddings[i] = char_glove.get(char)  # insert embedding
                else:
                    # TODO is initializing with zero vector smart?
                    char_embeddings[i] = np.zeros(Hyperparams.emb_size)
            self.char_embeddings = tf.Variable(tf.constant(char_embeddings), trainable=True, name="char_embeddings")

            print('start loading word embedding')
            word_embeddings = np.zeros([Hyperparams.word_vocab_size+1, Hyperparams.emb_size])
            word_glove = load_glove(Hyperparams.glove_word)  # load matrix
            for word, i in Hyperparams.word2id.items():
                if word in word_glove:
                    word_embeddings[i] = word_glove.get(word)  # insert embedding
                else:
                    # TODO is initializing with zero vector smart?
                    word_embeddings[i] = np.zeros(Hyperparams.emb_size)
            self.word_embeddings = tf.Variable(tf.constant(word_embeddings), trainable=False, name="word_embeddings")

        else:
            self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.char_vocab_size+1, Hyperparams.emb_size]),
                                               trainable=True, name="char_embeddings")
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.word_vocab_size+1, Hyperparams.emb_size]),
                                               trainable=False, name="word_embeddings")

        self.q_input_char = tf.placeholder(tf.int32, [Hyperparams.max_question_c], "question_c")
        self.q_input_word = tf.placeholder(tf.int32, [Hyperparams.max_question_w], "question_w")
        self.q_encoded_word, self.q_encoded_char = \
        encoding(self.q_input_word, self.q_input_char, self.word_embeddings, self.char_embeddings)

if __name__ == '__main__':
    sample_qc = convert_to_ids('What is in front of the Notre Dame Main Building?', mode='character')
    sample_qw = convert_to_ids('What is in front of the Notre Dame Main Building?', mode='word')

    QuACC = Model(load_pretrained=True)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {QuACC.q_input_char : sample_qc, QuACC.q_input_word : sample_qw}
        index = sess.run([QuACC.q_encoded_char, QuACC.q_encoded_word], feed_dict=feed_dict)

        print (index)
