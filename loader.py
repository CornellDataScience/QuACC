"""
Data loader class.
"""

import numpy as np
import os
import pandas as pd
import random
import spacy
from hyperparams import Hyperparams as Hp
from tqdm import tqdm

nlp = spacy.blank('en')


def tokenize(text, mode='word'):
    """Return list of tokenized words or characters.

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
        parsed = nlp(text)
        tokens = [i.text for i in parsed]
        return tokens


def convert_to_ids(text, ttype='question', mode='word'):
    """Return list of unique ids for the corresponding word or character in the input text.

    Args:
        text (str):  string to vectorize
        ttype (str): either 'question' or 'paragraph'
        mode (str):  'character' or 'word'. Change the scope of tokenization
    Returns:
        array of unique ids for the corresponding characters or words
    """
    assert mode in {'character', 'word'}, "Select 'word' or 'character'"
    assert ttype in {'question', 'paragraph'}, "Select 'question' or 'paragraph'"
    lookup = Hp.char2id if mode == 'character' else Hp.word2id

    option = [Hp.max_q_chars, Hp.max_q_words] if ttype == 'question' else [Hp.max_p_chars, Hp.max_p_words]
    max_len = option[0] if mode == 'character' else option[1]

    tokenized = tokenize(text, mode)
    ids = np.zeros(max_len).astype(np.int32)
    for i, c in enumerate(tokenized[:max_len]):
        ids[i] = int(lookup[c])
    return ids


def answer_pointers(answer, paragraph, ptr):
    """Find pointers to words corresponding to beginning and end of answer.

    Args:
        answer (str):    Answer to question
        paragraph (str): Relevant paragraph
        ptr (int):       Character pointer to start of answer
    Returns:
        list:            List of length 2: [word index of start, word index of end]
    """
    answer = tokenize(answer)
    n = len(tokenize(paragraph[:ptr]))
    answer_length = len(answer)
    return [n, n + answer_length]


class Loader(object):
    """Load & batch text data."""
    def __init__(self, batch_size, split=(0.8, 0.1, 0.1)):
        """Create loader.

        Args:
            batch_size (int):
            split (tuple):
        """
        self.text_data = pd.read_csv('./data/all_data.csv')
        self.batch_size = batch_size

        self.splits = None
        self.p_embeds, self.q_embeds = None, None
        self.batches_tr, self.batches_va, self.batches_te = {}, {}, {}
        self.ptr_tr, self.ptr_va, self.ptr_te = 0, 0, 0

        self.pre_process()
        self.create_batches(split)
        print('')

    def pre_process(self):
        """Pre-process data."""
        if os.path.exists('./data/paragraph_embed.npy'):
            self.p_embeds = np.load('./data/paragraph_embed.npy')
        else:
            self.p_embeds = []
            print('Generating paragraph embeddings...')
            for i in tqdm(range(self.text_data.shape[0])):
                paragraph = self.text_data.iloc[i]['Paragraph']
                self.p_embeds.append(convert_to_ids(paragraph, 'paragraph', 'word'))
            self.p_embeds = np.array(self.p_embeds)

        if os.path.exists('./data/question_embed.npy'):
            self.q_embeds = np.load('./data/question_embed.npy')
        else:
            self.q_embeds = []
            print('Generating question embeddings...')
            for i in tqdm(range(self.text_data.shape[0])):
                question = self.text_data.iloc[i]['Question']
                self.q_embeds.append(convert_to_ids(question, 'question', 'word'))
                self.q_embeds = np.array(self.q_embeds)

    def _assign_batch(self, indices):
        """Create a batch from list of indices."""
        batches = dict()
        batches['n_batches'] = n_batches = len(indices) // self.batch_size
        batches['p_embeds'] = np.split(self.p_embeds[indices, :], n_batches)
        batches['q_embeds'] = np.split(self.q_embeds[indices, :], n_batches)
        selected_rows = self.text_data.reindex(indices)
        batches['p_lengths'] = np.split(np.minimum(selected_rows['P_Length'].values, Hp.max_p_words), n_batches)
        batches['q_lengths'] = np.split(np.minimum(selected_rows['Q_Length'].values, Hp.max_q_words), n_batches)
        batches['pointers'] = np.split(selected_rows[['Start', 'End']].values, n_batches)
        return batches

    def create_batches(self, split):
        """Randomly shuffle data and split into training, validation & testing batches."""
        total_samples = int((self.text_data.shape[0] // (self.batch_size/min(split))) * (self.batch_size/min(split)))
        permutation = np.random.permutation(total_samples)
        sections = np.cumsum(total_samples * np.array(split)).astype(int)[:len(split) - 1]
        self.splits = np.split(permutation, sections)

        self.batches_tr = self._assign_batch(self.splits[0])
        if len(split) == 2:
            self.batches_te = self._assign_batch(self.splits[1])
        elif len(split) == 3:
            self.batches_va = self._assign_batch(self.splits[1])
            self.batches_te = self._assign_batch(self.splits[2])

    def shuffle_train_batches(self):
        """Shuffle batches."""
        random.shuffle(self.splits[0])
        self.batches_tr = self._assign_batch(self.splits[0])

    def next_training_batch(self):
        """Return current training batch; shuffle batches every epoch."""
        self.ptr_tr = (self.ptr_tr + 1) % self.batches_tr['n_batches']
        if self.ptr_tr == 0:
            self.shuffle_train_batches()
        batches, i = self.batches_tr, self.ptr_tr
        return batches['p_embeds'][i], batches['q_embeds'][i], batches['p_lengths'][i], batches['q_lengths'][i], \
            batches['pointers'][i]

    def next_validation_batch(self):
        """Return current validation batch."""
        self.ptr_va = (self.ptr_va + 1) % self.batches_va['n_batches']
        batches, i = self.batches_va, self.ptr_va
        return batches['p_embeds'][i], batches['q_embeds'][i], batches['p_lengths'][i], batches['q_lengths'][i], \
            batches['pointers'][i]

    def next_testing_batch(self):
        """Return current testing batch."""
        self.ptr_te = (self.ptr_te + 1) % self.batches_te['n_batches']
        batches, i = self.batches_te, self.ptr_te
        return batches['p_embeds'][i], batches['q_embeds'][i], batches['p_lengths'][i], batches['q_lengths'][i], \
            batches['pointers'][i]


if __name__ == '__main__':
    data = Loader(100)
