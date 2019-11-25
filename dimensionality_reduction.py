import argparse
import numpy as np
from sklearn import decomposition
from embedding import load_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embedding', required=True)
    parser.add_argument('-m', '--method', choices=['PCA', 'tSVD', 'DRA'], default='DRA')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-n', '--reduce_to', type=int, default=300)
    parser.add_argument('-b', '--do_in_batches', action='store_true')
    parser.add_argument('-nb', '--batch_size', type=int, default=1024)
    args = parser.parse_args()

    emb = load_embedding(args.embedding, lower=False, length_normalize=False, delete_duplicates=True)

    if args.method == 'PCA':
        if args.do_in_batches:
            emb.vectors = PPA_batches(emb.vectors, args.reduce_to, args.batch_size)
        else:
            emb.vectors = PCA(emb.vectors, args.reduce_to)

    elif args.method == 'tSVD':
        emb.vectors = T_SVD(emb.vectors, args.reduce_to)

    elif args.method == 'DRA':
        if args.do_in_batches:
            emb.vectors = DRA_batches(emb.vectors, args.reduce_to, args.batch_size)
        else:
            emb.vectors = DRA(emb.vectors, args.reduce_to)

    else:
        raise ValueError(str(args.method) + ' reduction method not supported. Reduction method supported: PCA, tSVD, DRA')

    emb.export(args.output)


def PPA(matrix):
    # PCA to get Top Components
    n = matrix.shape[1]
    pca = decomposition.PCA(n_components=n)
    X_train = matrix - np.mean(matrix)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(X_train):
        for u in U1[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        z.append(x)

    return np.asarray(z)


def PPA_batches(matrix, batch_size):
    # PCA to get Top Components
    n = matrix.shape[1]
    pca = decomposition.IncrementalPCA(n_components=n, batch_size=batch_size)
    X_train = matrix - np.mean(matrix)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(X_train):
        for u in U1[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        z.append(x)

    return np.asarray(z)


def PCA(matrix, n):
    pca = decomposition.PCA(n_components=n)
    # SKLEARN CENTERS DE DATA, THIS IS REDUNTDANT.
    X_train = matrix - np.mean(matrix)
    return pca.fit_transform(X_train)


def PCA_batches(matrix, n, batch_size):
    pca = decomposition.IncrementalPCA(n_components=n, batch_size=batch_size)
    # SKLEARN CENTERS DE DATA, THIS IS REDUNTDANT.
    X_train = matrix - np.mean(matrix)
    return pca.fit_transform(X_train)


def T_SVD(matrix,n):
    svd = decomposition.TruncatedSVD(n_components = n)
    return svd.fit_transform(matrix)


def DRA(matrix, n):
    return PPA(PCA(PPA(matrix), n))


def DRA_batches(matrix, n, batch_size):
    return PPA_batches(PCA_batches(PPA_batches(matrix, batch_size), n, batch_size), batch_size)


if __name__ == '__main__':
    main()
