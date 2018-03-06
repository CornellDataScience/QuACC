"""
Generate character embeddings from GloVe word embeddings.
Inspired by https://github.com/minimaxir/char-embeddings/blob/master/create_embeddings.py
"""

import argparse
import numpy as np
import os
import sys


def train(args):
    """Generates 300-dimensional character vectors. The character vectors are a weighted sum of all the word vectors
    that contain the given character."""

    char_vectors = {}                                         # each value is a tuple of (cumulative sum, # occurences)
    with open(args.file_path, 'r') as glove:
        for line in glove:
            line_split = line.strip().split(' ')              # remove leading/trailing whitespace, split on space
            word_vec = np.array(line_split[1:], dtype=float)  # 300-dimensional vector from GloVe
            word_txt = line_split[0]                          # actual word
            if len(word_vec) != 300:
                word_vec = np.array(line_split, dtype=float)
                word_txt = ' '
                print(word_vec.shape, line)

            for char in word_txt:
                if ord(char) < 128:                           # check if char is normal ASCII character
                    if char in char_vectors:
                        # add word vector to cumulative sum, increment counter
                        char_vectors[char] = (char_vectors[char][0] + word_vec, int(char_vectors[char][1]) + 1)
                    else:
                        # if first occurence of character, initialize sum to word vector and set counter to 1
                        char_vectors[char] = (word_vec, 1)

    output_file = os.path.join('models', os.path.splitext(os.path.basename(args.file_path))[0] + '-char.txt')
    with open(output_file, 'w') as char_embeds:
        for char in char_vectors:
            # average all word vectors containing character (& round to 6 decimal places)
            char_vec = np.round(char_vectors[char][0] / char_vectors[char][1], 6).tolist()
            char_embeds.write('{} {}\n'.format(char, ' '.join(str(x) for x in char_vec)))


def embed(args):
    # TODO: generate character embeddings for a given passage of text
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Train (generate char embeddings) or Embed (embed text).')
    parser.add_argument('file_path', type=str, help='Path to GloVe word embeddings or text to be embedded.')
    args = parser.parse_args(sys.argv[1:])

    if args.mode.lower() == 'train':
        train(args)
    elif args.mode.lower() == 'embed':
        embed(args)
