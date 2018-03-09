''' data loader with tf.train.batch class
'''
import spacy
from hyperparams import Hyperparams
import numpy as np


def tokenize(text, mode = 'character'):
    assert mode in {'character', 'word'}, "Select 'word' or 'character'"
    if mode == 'character':
        return [c for c in text]
    elif mode == 'word':
        nlp = spacy.blank('en')
        parsed = nlp(text)
        tokens = [i.text for i in parsed]
        return tokens

def convert_to_ids(text, mode = 'character'):
    assert mode in {'character', 'word'}, "Select 'word' or 'character'"
    lookup = Hyperparams.char2id if mode == 'character' else Hyperparams.word2id
    max_len = Hyperparams.max_question_c if mode == 'character' else Hyperparams.max_question_w
    tokenized = tokenize(text, mode)
    ids = (np.ones(max_len) * len(lookup)).astype(np.int32)
    for i, c in enumerate(tokenized[:max_len]):
        ids[i] = int(lookup[c])
    return ids


if __name__ == '__main__':
    print (convert_to_ids('What is in front of the Notre Dame Main Building?', mode = 'character'))
    print (convert_to_ids('What is in front of the Notre Dame Main Building?', mode = 'word'))
