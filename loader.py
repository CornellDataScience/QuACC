"""
Data loader class.
"""

import pandas as pd
import numpy as np
import spacy
from hyperparams import Hyperparams as Hp
from tqdm import tqdm
import os

nlp = spacy.blank('en')
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
    def __init__(self, batch_size, paragraphs=None, questions=None, save=True, load=False):
        # TODO: implement saving/loading
        self.raw_paragraphs = paragraphs or pd.read_csv('./data/raw_context.csv')
        self.raw_questions = questions or pd.read_csv('./data/raw_questions.csv')
        self.batch_size = batch_size
        self.n_batches = None

        self.paragraphs, self.questions = {}, []
        self.p_embeds, self.q_embeds, self.p_lengths, self.q_lengths, self.pointers = [], [], [], [], []
        self.p_batch, self.p_l_batch, self.q_batch, self.q_l_batch, self.ptr_batch = None, None, None, None, None
        self.pointer = 0

        self.pre_process(load)
        self.create_batches()

        print('Loaded {} paragraphs & {} questions.'.format(len(self.paragraphs), len(self.questions)))

    def pre_process(self, load = False):
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
            self.questions.append(question['Question'])
        if load:
            assert 'paragraph_embed.npy' in os.listdir('./data'), "paragraph_embed.npy is missing"
            assert 'question_embed.npy' in os.listdir('./data'), "question_embed.npy is missing"
            assert 'paragraph_length.npy' in os.listdir('./data'), "paragraph_length.npy is missing"
            assert 'question_lengths.npy' in os.listdir('./data'), "question_lengths.npy is missing"
            assert 'word_pointers.npy' in os.listdir('./data'), "word_pointers.npy is missing"
            print('Loading matrices...')
            self.p_embeds = np.load('./data/paragraph_embed.npy')
            self.q_embeds = np.load('./data/question_embed.npy')
            self.p_lengths = np.load('./data/paragraph_length.npy')
            self.q_lengths = np.load('./data/question_lengths.npy')
            self.pointers = np.load('./data/word_pointers.npy')

        else:
            # pre_processed_pointers = None
            # if 'questions.csv' in os.listdir('data'):
            #     pre_processed_pointers = pd.read_csv('./data/questions.csv')
            print('Processing Questions...')
            for i in tqdm(range(self.raw_questions.shape[0])):
                question = self.raw_questions.iloc[i]
                answer = question['Answer']
                paragraph = self.paragraphs[question['Topic']][question['Paragraph #']]
                char_ptr = question['Pointer']
                pointers = answer_pointers(answer, paragraph, char_ptr)
                self.p_embeds.append(convert_to_ids(paragraph, 'paragraph'))
                self.q_embeds.append(convert_to_ids(question['Question'], 'question'))
                self.p_lengths.append(len(tokenize(paragraph)))
                self.q_lengths.append(len(tokenize(question['Question'])))
                self.pointers.append(pointers)

                if (i % 1000) == 0:
                    print('Saving first {} pointers...'.format(i))
                    np.save('./data/word_pointers', np.array(self.pointers).astype(int))

            np.save('./data/paragraph_embed', np.array(self.p_embeds).astype(int))
            np.save('./data/question_embed', np.array(self.q_embeds).astype(int))
            np.save('./data/paragraph_length', np.array(self.p_lengths).astype(int))
            np.save('./data/question_lengths', np.array(self.q_lengths).astype(int))
            np.save('./data/word_pointers', np.array(self.pointers).astype(int))

            # if pre_processed_pointers is None:
            #     pointers_df = pd.DataFrame(np.array(self.pointers).astype(int), columns=['Start', 'End'])
            #     combined = pd.concat([self.raw_questions, pointers_df], axis=1)[['Topic', 'Paragraph #', 'Question', 'Answer', 'Pointer', 'Start', 'End']]
            #     combined.to_csv('./data/questions.csv', index=False)

    def create_batches(self):
        """Randomly shuffle data and split into training batches."""
        self.n_batches = int(len(self.questions) / self.batch_size)
        # truncate training data so it is equally divisible into batches
        n_samples = self.n_batches * self.batch_size
        permutation = np.random.permutation(n_samples)

        p_embeds_b = np.array(self.p_embeds)[:n_samples, :]
        q_embeds_b = np.array(self.q_embeds)[:n_samples, :]
        p_lengths_b = np.array(self.p_lengths)[:n_samples]
        q_lengths_b = np.array(self.q_lengths)[:n_samples]
        pointers_b = np.array(self.pointers)[:n_samples, :]

        # split training data into equally sized batches
        self.p_batch = np.split(p_embeds_b[permutation, :], self.n_batches, 0)
        self.q_batch = np.split(q_embeds_b[permutation, :], self.n_batches, 0)
        self.p_l_batch = np.split(p_lengths_b[permutation], self.n_batches)
        self.q_l_batch = np.split(q_lengths_b[permutation], self.n_batches)
        self.ptr_batch = np.split(pointers_b[permutation, :], self.n_batches, 0)

    def next_batch(self):
        """Return current batch; shuffle batches every epoch."""
        self.pointer = (self.pointer + 1) % self.n_batches
        if self.pointer == 0:
            self.create_batches()
        return self.p_batch[self.pointer], self.q_batch[self.pointer], self.p_l_batch[self.pointer], \
            self.q_l_batch[self.pointer], self.ptr_batch[self.pointer]


if __name__ == '__main__':
    l = Loader(100, load=True)
