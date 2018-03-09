"""
All hyperparameters. Implemented as a class for convenience, but operates identically to an argument parser.
"""
import pickle


class Hyperparams:

    # pre-trained models
    glove_word = './models/glove.840B.300d.txt'
    glove_char = './models/glove.840B.300d-char.txt'

    # data
    data_dir = './data'

    # training
    dropout = 0.2
    optimizer = 'adam'

    # architecture
    emb_size = 300

    # SQuAD related info
    with open('./data/question-word-dict.pkl', 'rb') as f:
        question_word_dict = pickle.load(f)
    with open('./data/question-char-dict.pkl', 'rb') as f:
        question_char_dict = pickle.load(f)

    # word vocabulary
    vocab_size = len(question_word_dict)
    # character vocabulary
    char_vocab_size = len(question_char_dict)
