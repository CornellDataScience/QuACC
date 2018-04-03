"""
Pre-process training data.
"""

import argparse
import json
import pandas as pd
import os
import spacy
import sys
import time
from loader import tokenize


def generate_dict(save=True):
    # make sure the file exists
    assert 'raw_questions.csv' in os.listdir('data'), "Load Question Data First"
    assert 'raw_context.csv' in os.listdir('data'), "Load Context Data First"
    # load dataframe
    rq = pd.read_csv('data/raw_questions.csv', index_col=False)
    rc = pd.read_csv('data/raw_context.csv', index_col=False)
    # array to text
    a_text = ' '.join(rq['Answer'].values.astype(str))
    q_text = ' '.join(rq['Question'].values.astype(str))
    c_text = ' '.join(rc['Context'].values.astype(str))
    # generate char dict
    c = set(tokenize(a_text, 'character'))
    c.update(set(tokenize(q_text, 'character')))
    c.update(set(tokenize(c_text, 'character')))
    # generate word dict
    w = set(tokenize(a_text, 'word'))
    w.update(set(tokenize(q_text, 'word')))
    w.update(set(tokenize(c_text, 'word')))
    # save
    if save:
        char2id = {char: i+2 for i, char in enumerate(c)}
        id2char = {i+2: char for i, char in enumerate(c)}
        word2id = {word: i+2 for i, word in enumerate(w)}
        id2word = {i+2: word for i, word in enumerate(w)}
        with open('data/char2id.json', 'w') as file:
            json.dump(char2id, file)
        with open('data/id2char.json', 'w') as file:
            json.dump(id2char, file)
        with open('data/word2id.json', 'w') as file:
            json.dump(word2id, file)
        with open('data/id2word.json', 'w') as file:
            json.dump(id2word, file)


def process_context(data, save=True):
    all_context = []
    for topic in data:
        article = topic['title']
        for i, paragraph in enumerate(topic['paragraphs']):
            context = paragraph['context']
            all_context.append([article, i, context])
    headings = ['Topic', 'Paragraph #', 'Context']
    dataframe = pd.DataFrame(all_context, columns=headings)
    if save:
        dataframe.to_csv('data/raw_paragraphs.csv', index=False)
    return dataframe


def process(data, save=True):
    all_questions = []
    for topic in data:
        article = topic['title']
        for i, paragraph in enumerate(topic['paragraphs']):
            for qa in paragraph['qas']:
                question = qa['question']
                for ans in qa['answers']:
                    answer = ans['text']
                    answer_idx = ans['answer_start']
                    all_questions.append([article, i, question, answer, answer_idx])
    headings = ['Topic', 'Paragraph #', 'Question', 'Answer', 'Pointer']
    dataframe = pd.DataFrame(all_questions, columns=headings)
    if save:
        dataframe.to_csv('data/raw_questions.csv', index=False)
    return dataframe


def tokenize_spacy(docs):
    """Tokenize a list of documents using spacy."""
    nlp = spacy.load('en')
    t0 = time.time()
    parsed = []
    # TODO: parallelize for loop
    for i, doc in enumerate(docs):
        parsed.append(nlp(str(doc)))
        if (i + 1) % 100 == 0:
            print('Processed {} answers'.format(i + 1))
    t1 = time.time() - t0
    print('\nProcessed {} answers in {:.2f}s'.format(len(docs), t1))

    with open('data/parsed_answers.json', 'w') as file:
        json.dump(parsed, file)


def main(args):
    with open(args.file, 'r') as file:
        data = json.load(file)['data']
    if 'raw_questions.csv' not in os.listdir('data'):
        dataframe = process(data, save=True)
    else:
        dataframe = pd.read_csv('data/raw_questions.csv')

    if 'raw_context.csv' not in os.listdir('data'):
        dataframe = process_context(data, save=True)
    else:
        dataframe = pd.read_csv('data/raw_paragraphs.csv')

    generate_dict()
    # tokenize(dataframe.loc[:, 'Answer'].values)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/train-v1.1.json', help='path to SQuAD data JSON file')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
