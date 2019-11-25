import numpy as np
import datetime
import argparse
from collections import defaultdict
import sys
sys.path.insert(0, '../')
from tensorflow_functions import matrix_add, cosine_knn_batches, cosine_knn
from embedding import load_embedding
from utils import get_dimensions
from utils import vocab_from_path, normalize_vector, printTrace, batch



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embedding', required=True)
    parser.add_argument('-c', '--cross_embedding', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-k', '--num_nearest_neighbor', type=int, default=10)


    args = parser.parse_args()


    dims = get_dimensions(args.embedding)

    if dims != get_dimensions(args.cross_embedding):
        raise ValueError('All the embeddings must have the same number of dimensions and the embeddings must be in the word2vec format')

    printTrace('Reading vocab...')

    vocab_emb = vocab_from_path(args.embedding)
    vocab_cross = vocab_from_path(args.cross_embedding)

    total_vocab = set.union(set(vocab_emb), set(vocab_cross))
    interset_vocab = list(set.intersection(set(vocab_emb), set(vocab_cross)))
    vocab_to_generate = set(vocab_cross) - set(vocab_emb)

    print('Final embedding will have ' + str(len(total_vocab)) + ' words')
    print('We will generate ' + str(len(vocab_to_generate)) + ' words')

    emb = load_embedding(args.cross_embedding, vocabulary=None, lower=False, length_normalize=True, normalize_dimensionwise=False,
                         delete_duplicates=True)

    m = emb.words_to_matrix(vocab_to_generate)

    M = emb.words_to_matrix(interset_vocab)

    nn=[]

    for i_batch, mb in enumerate(batch(m, args.batch_size)):

        string = "<" + str(datetime.datetime.now()) + ">  " + 'Using Embedding ' + str(
            args.cross_embedding) + ' to generate vocab for Embedding ' + str(args.embedding) + ':  ' + str(
            int(100 * (i_batch * args.batch_size) / len(m))) + '%'
        print(string, end="\r")

        # print(np.asarray(mb).shape)
        # print(np.asarray(M).shape)


        result = cosine_knn(mb, M, args.num_nearest_neighbor)

        for i_result, indexes in enumerate(result):
            nn.append([interset_vocab[i] for i in indexes])

    del emb


    printTrace('===> Generating new_vocab <===')

    emb = load_embedding(args.embedding, vocabulary=None, lower=False, length_normalize=False, normalize_dimensionwise=False,
                         delete_duplicates=True)



    new_vectors = []
    for i_word, word in enumerate(vocab_to_generate):
        if i_word%1000 == 0:
            string = "<" + str(datetime.datetime.now()) + ">  " + 'Generating vocab ' + args.output + ': ' + str(
                int(100 * i_word / len(vocab_to_generate))) + '%'
            print(string, end="\r")

        try:
            lw = nn[i_word]
            v = np.zeros([dims], dtype=float)
            for word_nn in lw:
                v += emb.word_to_vector(word_nn)

        except KeyError as r:
            raise ValueError('Something went wrong in the word generation process')

        new_vectors.append(v/args.num_nearest_neighbor)

    print()


    printTrace('===> Printing to file <===')

    with open(args.output,'w') as file:

        print(str(len(emb.words)+len(vocab_to_generate)) + ' ' + str(dims),file=file)

        for w in emb.words:
            print(w + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(w)]), file=file)

        for w_i, w in enumerate(vocab_to_generate):
            print(w + ' ' + ' '.join(['%.6g' % x for x in new_vectors[w_i]]), file=file)

if __name__ == '__main__':
    main()
