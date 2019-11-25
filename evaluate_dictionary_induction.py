from embedding import load_embedding
from tensorflow_functions import cosine_knn_batches
import argparse
from utils import printTrace, batch
import os
import numpy as np
import datetime

def main():
    parser = argparse.ArgumentParser()
    inputtype = parser.add_mutually_exclusive_group(required=True)
    inputtype.add_argument('-i', '--embedding', type=str)
    inputtype.add_argument('-d', '--directory', type=str)

    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-dic', '--dictionary_path', type=str, default='DictionaryInductionDataset/es-en.test')
    parser.add_argument('-p', '--add_lang_prefix', action='store_true')

    args = parser.parse_args()

    emb_list = []
    if args.embedding is not None:
        emb_list.append(args.embedding)
    else:
        emb_list = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if
                 os.path.isfile(os.path.join(args.directory, f))]

    if not os.path.exists('Results'):
        os.makedirs('Results')

    for emb_i, emb_path in enumerate(emb_list):

        printTrace('Evaluating Embedding ' + str(emb_i+1) + ' of ' + str(len(emb_list)) + ' : ' + str(emb_path))
        emb = load_embedding(emb_path, lower=False, length_normalize=True, delete_duplicates=True)

        top1, top2, top3, top5, top10, coverage = evaluate_dictionary_induction(emb, args.dictionary_path, args.batch_size, emb_path, args.add_lang_prefix)



        with open('Results/dictionary_induction', 'a+') as file:
            print(','.join([str(emb_path), str(top1), str(top2), str(top3), str(top5), str(top10), str(coverage)]), file=file)



    print('Results have been exported in csv format to the Results folder')


def evaluate_dictionary_induction(emb, dictionary_path, batch_size, name='nameless', add_prefix=False):

    oov = 0
    notfound = 0
    source_vectors = []
    target_words = []
    positions = []

    for line in open(dictionary_path, 'r', encoding='utf8'):
        try:
            w1, w2 = line.split()
            if add_prefix:
                a = emb.word_to_vector('es/'+w1)
                b = emb.word_to_vector('en/'+w2)
            else:
                a = emb.word_to_vector(w1)
                b = emb.word_to_vector(w2)

            source_vectors.append(a)
            if add_prefix:
                target_words.append('en/'+w2)
            else:
                target_words.append(w2)

        except KeyError:
            oov+=1

    for i_batch, matrix in enumerate(batch(source_vectors, batch_size)):

        string = "<" + str(datetime.datetime.now()) + ">  " + 'Evaluating dictionary induction for embedding ' + str(name) + ': ' + str(
            int(100 * (i_batch * batch_size) / len(target_words))) + '%'
        print(string, end="\r")

        top = cosine_knn_batches(matrix, emb.vectors, 10)

        for i_result, results in enumerate(top):

            try:
                positions.append(np.where(results == emb.word_to_index(target_words[i_batch*batch_size+i_result]))[0][0])
            except IndexError:
                notfound += 1

    positions = np.asarray(positions)
    top1 = len(np.where(positions < 1)[0]) / (len(positions) + notfound)
    top2 = len(np.where(positions < 2)[0]) / (len(positions) + notfound)
    top3 = len(np.where(positions < 3)[0]) / (len(positions) + notfound)
    top5 = len(np.where(positions < 5)[0]) / (len(positions) + notfound)
    top10 = len(positions) / (len(positions) + notfound)

    coverage = len(target_words) / (len(target_words) + oov)

    print('Embedding ' + str(name) + ' T@1=' + str(top1) + ', T@2=' + str(top2) + ', T@3=' + str(top3) + ', T@5='
          + str(top5) + ', T@10=' + str(top10) + '. Coverage=' + str(coverage))

    return top1, top2, top3, top5, top10, coverage


if __name__ == '__main__':
    main()
