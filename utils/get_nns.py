import sys
import datetime
import argparse
sys.path.insert(0, '../')
from tensorflow_functions import cosine_knn
from embedding import load_embedding
from utils import vocab_from_path, printTrace, batch

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embedding', required=True)
    parser.add_argument('-l', '--search_words', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-k', '--num_nearest_neighbor', type=int, default=10)

    args = parser.parse_args()

    emb = load_embedding(args.embedding, vocabulary=None, lower=False, length_normalize=True,
                         normalize_dimensionwise=False,
                         delete_duplicates=True)

    words_2_search = vocab_from_path(args.search_words)

    m = emb.words_to_matrix(words_2_search)
    M = emb.words_to_matrix(emb.words)

    nn = []

    for i_batch, mb in enumerate(batch(m, args.batch_size)):

        string = "<" + str(datetime.datetime.now()) + ">  " + 'Calculating nn words  ' + str(
            int(100 * (i_batch * args.batch_size) / len(m))) + '%'
        print(string, end="\r")

        result = cosine_knn(mb, M, args.num_nearest_neighbor)

        for i_result, indexes in enumerate(result):
            nn.append(["\""+emb.words[i] + "\"" for i in indexes])

    file = open(args.output,'w+',encoding='utf-8')

    for word,nns in zip(words_2_search,nn):
        print(word + ': ' + ' '.join(nns), file=file)


if __name__ == '__main__':
    main()

