# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
import os
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec


def txt2bin(filename):
    m = gensim.models.KeyedVectors.load_word2vec_format(filename)
    m.save_word2vec_format(filename.replace('txt', 'bin'), binary=True)
    gensim.models.KeyedVectors.load(filename.replace('txt', 'bin'), mmap='r')


def generate(path):
    glove_vector_file = os.path.join(path, 'glove.6B.300d.txt')
    output_model_file = os.path.join(path, 'glove.6B.300d.model.txt')

    txt2bin(glove2word2vec(glove_vector_file, output_model_file))


if __name__ == "__main__":
    generate()
