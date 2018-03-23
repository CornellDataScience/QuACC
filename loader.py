"""
Data loader with tf.train.batch class
"""

import numpy as np
import spacy
from hyperparams import Hyperparams


def tokenize(text, mode='character'):
    """return list of tokenized word or character
    Args:
        text (string): string to vectorize
        mode (string): 'character' or 'word'. Change the scope of tokenization
    Returns:
        array of tokens
    """
    assert mode in {'character', 'word'}, "Select 'word' or 'character'"
    if mode == 'character':
        return [c for c in text]
    elif mode == 'word':
        nlp = spacy.blank('en')
        parsed = nlp(text)
        tokens = [i.text for i in parsed]
        return tokens


def convert_to_ids(text, ttype = 'question', mode='character'):
    """return list of unique ids for the corresponding word or character in the input text
    Args:
        text (string): string to vectorize
        mode (string): 'character' or 'word'. Change the scope of tokenization
    Returns:
        array of unique ids for the corresponding characters or words
    """
    assert mode in {'character', 'word'}, "Select 'word' or 'character'"
    assert ttype in {'question', 'context'}, "Select 'question' or 'context'"
    lookup = Hyperparams.char2id if mode == 'character' else Hyperparams.word2id

    option = [Hyperparams.max_question_c, Hyperparams.max_question_w] if ttype == 'question'\
            else [Hyperparams.max_context_c, Hyperparams.max_context_w]
    max_len = option[0] if mode == 'character' else option[1]
    
    tokenized = tokenize(text, mode)
    ids = (np.ones(max_len) * len(lookup)).astype(np.int32)
    for i, c in enumerate(tokenized[:max_len]):
        ids[i] = int(lookup[c])
    return ids


if __name__ == '__main__':
    print(convert_to_ids('What is in front of the Notre Dame Main Building?', mode='character'))
    print(convert_to_ids('What is in front of the Notre Dame Main Building?', mode='word'))
