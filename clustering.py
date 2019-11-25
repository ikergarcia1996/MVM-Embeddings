from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from embedding import load_embedding
from utils import printTrace
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    inputtype = parser.add_mutually_exclusive_group(required=True)
    inputtype.add_argument('-i', '--embedding', type=str)
    inputtype.add_argument('-d', '--directory', type=str)
    parser.add_argument('-k', '--num_clusters', type=int, required=True)
    parser.add_argument('-m', '--metric', type=str, default='cosine')
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    args = parser.parse_args()

    if args.embedding is not None:
        output = ''.join(args.embedding.split('/')[-1].split('.')[:-1]) + '_' + str(args.metric) + '_' + str(args.num_clusters) + '.clusters' if args.output is None else args.output

        kmeans4embedding(args.embedding, output, args.num_clusters, args.metric, args.batch_size)

    else:
        files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f))]
        for i_file, file in enumerate(files):
            printTrace('==> Doing clustering for embedding ' + str(i_file) + ' of ' + str(len(files)) + ' : ' + str(file))

            if args.output is None:
                if not os.path.exists('Clustering'):
                    os.makedirs('Clustering')

            output = str(('Clustering/' if args.output is None else args.output)) + ''.join(file.split('/')[-1].split('.')[:-1]) + '_' + str(args.metric) + '_' + str(args.num_clusters) + '.clusters'

            kmeans4embedding(file, output, args.num_clusters, args.metric, args.batch_size)


def kmeans4embedding(embedding_path, output_path, k, metric, batch_size):
    printTrace('Loading embedding ' + str(embedding_path))

    emb = load_embedding(embedding_path, lower=False, length_normalize=False, delete_duplicates=True)

    printTrace('Clustering for embedding ' + str(embedding_path))

    labels = doKmeans(emb.vectors, k, metric, batch_size)

    printTrace('Printing clusters for embedding ' + str(embedding_path))

    with open(output_path, 'w') as file:
        for i_label, label in enumerate(labels):
            print(emb.vocabulary.index_to_word(i_label) + ' ' + str(label), file=file)

    printTrace('Sorting clusters for embedding ' + str(embedding_path))

    excec_com = 'sort -k2 -n ' + str(output_path) + ' > ' + str(output_path) + '_sorted'
    print(excec_com)
    os.system(excec_com)
    excec_com = 'rm ' + str(output_path)
    print(excec_com)
    os.system(excec_com)
    excec_com = 'mv ' + str(output_path) + '_sorted ' + str(output_path)
    print(excec_com)
    os.system(excec_com)

    printTrace('Done, clusters saved in ' + str(output_path))


def doKmeans(matrix, k, metric='cosine', batch_size=1024):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)

    mbk.fit(matrix)

    return pairwise_distances_argmin(X=matrix, Y=mbk.cluster_centers_, metric=metric)


if __name__ == '__main__':
    main()
