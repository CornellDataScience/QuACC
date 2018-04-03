"""
Data loader class.
"""

import pandas as pd
import numpy as np
import spacy
from hyperparams import Hyperparams as Hp
from tqdm import tqdm


def tokenize(text, mode='word'):
    """Return list of tokenized worsd or characters.

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
    paragraph = tokenize(paragraph[ptr:])
    answer_length = len(answer)
    for start in (i for i, word in enumerate(paragraph) if word == answer[0]):
        if paragraph[start: start + answer_length] == answer:
            return [n + start, n + start + answer_length]


class Loader(object):
    """Load & batch text data."""
    def __init__(self, batch_size, paragraphs=None, questions=None, save=True, load=False):
        self.raw_paragraphs = paragraphs or pd.read_csv('./data/raw_paragraphs.csv')
        self.raw_questions = questions or pd.read_csv('./data/raw_questions.csv')
        self.batch_size = batch_size
        self.n_batches = None

        self.paragraphs, self.questions = {}, []
        self.p_embeds, self.q_embeds, self.p_lengths, self.q_lengths, self.pointers = [], [], [], [], []
        self.p_batch, self.p_l_batch, self.q_batch, self.q_l_batch, self.ptr_batch = None, None, None, None, None
        self.pointer = 0

        self.pre_process()
        self.create_batches()
        print('Loaded {} paragraphs & {} questions.'.format(len(self.paragraphs), len(self.questions)))

    def pre_process(self):
        """Pre-process data."""
        print('Processing paragraphs...')
        for i in tqdm(range(self.raw_paragraphs.shape[0])):
            topic = self.raw_paragraphs.iloc[i]['Topic']
            context = self.raw_paragraphs.iloc[i]['Context']
            if topic in self.paragraphs.keys():
                self.paragraphs[topic].append(context)
            else:
                self.paragraphs[topic] = [context]
        print('Processing Questions...')
        for i in tqdm(range(self.raw_questions.shape[0])):
            question = self.raw_questions.iloc[i]
            answer = question['Answer']
            paragraph = self.paragraphs[question['Topic']][question['Paragraph #']]
            char_ptr = question['Pointer']
            pointers = answer_pointers(answer, paragraph, char_ptr)
            self.questions.append(question['Question'])
            self.p_embeds.append(convert_to_ids(paragraph, 'paragraph'))
            self.q_embeds.append(convert_to_ids(question['Question'], 'question'))
            self.p_lengths.append(len(tokenize(paragraph)))
            self.q_lengths.append(len(tokenize(question['Question'])))
            self.pointers.append(pointers)

    def create_batches(self):
        """Randomly shuffle data and split into training batches."""
        self.n_batches = int(len(self.questions) / self.batch_size)
        # truncate training data so it is equally divisible into batches
        n_samples = self.n_batches * self.batch_size
        permutation = np.random.permutation(n_samples)
        self.p_embeds = np.array(self.p_embeds)[:n_samples, :]
        self.q_embeds = np.array(self.q_embeds)[:n_samples, :]
        self.p_lengths = np.array(self.p_lengths)[:n_samples]
        self.q_lengths = np.array(self.q_lengths)[:n_samples]
        self.pointers = np.array(self.pointers)[:n_samples, :]

        # split training data into equally sized batches
        self.p_batch = np.split(self.p_embeds[permutation, :], self.n_batches, 0)
        self.q_batch = np.split(self.q_embeds[permutation, :], self.n_batches, 0)
        self.p_l_batch = np.split(self.p_lengths[permutation, :], self.n_batches)
        self.q_l_batch = np.split(self.q_lengths[permutation, :], self.n_batches)
        self.ptr_batch = np.split(self.pointers[permutation, :], self.n_batches, 0)

    def next_batch(self):
        """Return current batch; shuffle batches every epoch."""
        self.pointer = (self.pointer + 1) % self.n_batches
        if self.pointer == 0:
            self.create_batches()
        return self.p_batch[self.pointer], self.q_batch[self.pointer], self.p_l_batch[self.pointer], \
            self.q_l_batch[self.pointer], self.ptr_batch[self.pointer]


if __name__ == '__main__':
    print(convert_to_ids('What is in front of the Notre Dame Main Building?', mode='character'))
    print(convert_to_ids('What is in front of the Notre Dame Main Building?', mode='word'))
