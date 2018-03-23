"""
Pre-process training data.
"""

import argparse
import json
import pandas as pd
import pickle
import os
import spacy
import sys
import time
import loader


def generate_dict(save = True):
    # make sure the file exists
    assert ['raw_questions.csv'] not in os.listdir('data'), "Load Question Data First"
    assert ['raw_context.csv'] not in os.listdir('data'), "Load Context Data First"
    # load datafram
    rq = pd.read_csv('data/raw_questions.csv', index_col=False)
    rc = pd.read_csv('data/raw_context.csv', index_col=False)
    # array to text
    a_text = ' '.join(rq['Answer'].values)
    q_text = ' '.join(rq['Question'].values)
    c_text = ' '.join(rc['Context'].values)
    # generate char dict
    c = set(loader.tokenize(a_text, 'character'))
    c.update(set(loader.tokenize(q_text, 'character')))
    c.update(set(loader.tokenize(c_text, 'character')))
    # generate word dict
    w = set(loader.tokenize(a_text, 'word'))
    w.update(set(loader.tokenize(q_text, 'word')))
    w.update(set(loader.tokenize(c_text, 'word')))
    # save
    if save:
        char2id = {ch : i for i, ch in enumerate(c)}
        id2char = {i : ch for i, ch in enumerate(c)}
        word2id = {wd : i for i, wd in enumerate(w)}
        id2word = {i : wd for i, wd in enumerate(w)}
        with open('data/char2id-dict.pkl', 'wb') as d:
            pickle.dump(char2id, d)
        with open('data/id2char-dict.pkl', 'wb') as d:
            pickle.dump(id2char, d)
        with open('data/word2id-dict.pkl', 'wb') as d:
            pickle.dump(word2id, d)
        with open('data/id2word-dict.pkl', 'wb') as d:
            pickle.dump(id2word, d)


def process_context(data, save = True):
    all_context = []
    for topic in data:
        article = topic['title']
        for i, paragraph in enumerate(topic['paragraphs']):
            context = paragraph['context']
            all_context.append([article, i, context])
    headings = ['Topic', 'Paragraph #', 'Context']
    dataframe = pd.DataFrame(all_context, columns=headings)
    if save:
        dataframe.to_csv('data/raw_context.csv', index=False)
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


def tokenize(docs):
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

    with open('data/parsed_answers.pkl', 'wb') as f:
        pickle.dump(parsed, f)


def main(args):
    with open(args.file, 'r') as file:
        data = json.load(file)['data']
    if ['raw_questions.csv'] not in os.listdir('data'):
        dataframe = process(data, save=True)
    else:
        dataframe = pd.read_csv('data/raw_questions.csv')

    if ['raw_context.csv'] not in os.listdir('data'):
        dataframe = process_context(data, save=True)
    else:
        dataframe = pd.read_csv('data/raw_context.csv')

    generate_dict()

    # tokenize(dataframe.loc[:, 'Answer'].values)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='data/train-v1.1.json', help='path to SQuAD training data JSON file')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
