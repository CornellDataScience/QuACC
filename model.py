"""
Main model.
"""

from hyperparams import Hyperparams
import tensorflow as tf
from layers import *
from util import load_glove


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
        char2id (dict):                 dictionary that maps a character to its unique id for the lookup
        word2id (dict):                 dictionary that maps a word to its unique id for the lookup
        id2char (dict):                 dictionary that maps an id to its corresponding character
        id2word (dict):                 dictionary that maps an id to its corresponding word
    """
    def __init__(self, load_pretrained=True):
        if load_pretrained:
            print('start loading character embedding')
            char_embeddings = np.zeros([Hyperparams.char_vocab_size, Hyperparams.emb_size])
            char_glove = load_glove(Hyperparams.glove_char)  # load matrix
            char2id = {}
            id2char = {}
            for i, char in enumerate(Hyperparams.question_char_dict):
                char2id[char] = i
                id2char[i] = char
                if char in char_glove.keys():
                    char_embeddings[i] = char_glove.get(char)  # insert embedding
                else:
                    # TODO is initializing with zero vector smart?
                    char_embeddings[i] = np.zeros(Hyperparams.emb_size)
            self.char_embeddings = tf.Variable(tf.constant(char_embeddings), trainable=True, name="char_embeddings")
            self.char2id = char2id
            self.id2char = id2char

            print('start loading word embedding')
            word_embeddings = np.zeros([Hyperparams.word_vocab_size, Hyperparams.emb_size])
            word_glove = load_glove(Hyperparams.glove_word)  # load matrix
            word2id = {}
            id2word = {}
            for i, word in enumerate(Hyperparams.question_word_dict):
                word2id[word] = i
                id2word[i] = word
                if word in word_glove.keys():
                    word_embeddings[i] = word_glove.get(word)  # insert embedding
                else:
                    # TODO is initializing with zero vector smart?
                    word_embeddings[i] = np.zeros(Hyperparams.emb_size)
            self.word_embeddings = tf.Variable(tf.constant(word_embeddings), trainable=False, name="word_embeddings")
            self.word2id = word2id
            self.id2word = id2word

        else:
            self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.char_vocab_size, Hyperparams.emb_size]),
                                               trainable=True, name="char_embeddings")
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.word_vocab_size, Hyperparams.emb_size]),
                                               trainable=False, name="word_embeddings")


if __name__ == '__main__':
    QuACC = Model()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # print(QuACC.char_embeddings.eval())
        # print(QuACC.char2id)
        # print(QuACC.word_embeddings.eval())
