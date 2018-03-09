"""
This is a model
"""
from hyperparams import Hyperparams
import tensorflow as tf
import numpy as np
from layers import *


class Model(object):
    def __init__(self, load_file=True, load_pretrained=True):
        """Main model

        Attributes:
            load_pretrained  (bool): If True, load the model at the instantiation
        """
        if load_pretrained:
            char_embeddings = np.zeros((len(Hyperparams.question_char_dict), Hyperparams.emb_size))
            char_glove = self.load_glove(Hyperparams.glove_char)
            char2id = {}
            id2char = {}
            for i, char in enumerate(Hyperparams.question_char_dict):
                char2id[char] = i
                id2char[i] = char
                if char in char_glove.keys():
                    char_embeddings[i] = char_glove.get(char)
                else:
                    char_embeddings[i] = np.zeros(Hyperparams.emb_size)
            self.char_embeddings = tf.Variable(tf.constant(char_embeddings), trainable=True, name="char_embeddings")

            word_embeddings = np.zeros((len(Hyperparams.question_word_dict), Hyperparams.emb_size))
            word_glove = self.load_glove(Hyperparams.glove_word)
            word2id = {}
            id2word = {}
            for i, word in enumerate(Hyperparams.question_word_dict):
                word2id[word] = i
                id2word[i] = word
                if word in word_glove.keys():
                    word_embeddings[i] = word_glove.get(word)
                else:
                    word_embeddings[i] = np.zeros(Hyperparams.emb_size)

            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.vocab_size, Hyperparams.emb_size]),trainable=False, name="word_embeddings")
            self.char2id = char2id
            self.id2char = id2char

        else:
            self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.char_vocab_size, Hyperparams.emb_size]),trainable=True, name="char_embeddings")
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Hyperparams.vocab_size, Hyperparams.emb_size]),trainable=False, name="word_embeddings")


    def load_glove(self, directory):
        with open(directory, 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:], dtype=float)
                char = line_split[0]
                embedding_vectors[char] = vec
        return embedding_vectors



if __name__ == '__main__':
    QuACC = Model()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(QuACC.char_embeddings.eval())
