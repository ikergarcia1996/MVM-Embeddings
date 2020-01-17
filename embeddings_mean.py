import numpy as np
from collections import defaultdict
from tensorflow_functions import matrix_add, cosine_knn
from embedding import load_embedding
from utils import get_dimensions
from utils import vocab_from_path, normalize_vector, printTrace, batch
import datetime
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--embeddings", nargs="+", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--vocabulary", default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=1024)
    parser.add_argument("-k", "--num_nearest_neighbor", type=int, default=10)
    parser.add_argument("-oov", "--generate_oov_words", action="store_false")

    args = parser.parse_args()
    if args.generate_oov_words:
        average_embeddings_generate(
            embeddings_path=args.embeddings,
            out_path=args.output,
            vocab=vocab_from_path(args.vocabulary) if args.vocabulary else None,
            batch_size=args.batch_size,
            k=args.num_nearest_neighbor,
        )
    else:
        average_embeddings(
            embeddings_path=args.embeddings,
            out_path=args.output,
            vocab=vocab_from_path(args.vocabulary) if args.vocabulary else None,
        )


def average_embeddings_generate(
    embeddings_path, out_path, vocab=None, batch_size=1024, k=10
):

    dims = 0
    for p in embeddings_path:
        if dims:

            d = get_dimensions(p)
            if d is None:
                raise ValueError("The embeddings must be in the word2vec format")

            elif dims == get_dimensions(p):
                continue

            else:
                raise ValueError(
                    "All the embeddings must have the same number of dimensions"
                    " and the embeddings must be in the word2vec format"
                )
        else:
            dims = get_dimensions(p)
            if dims is None:
                raise ValueError("The embeddings must be in the word2vec format")

    printTrace("Reading vocab...")

    # [[vocab_emb1], [vocab_emb_2], ...]
    vocab_embeddings = [vocab_from_path(x) for x in embeddings_path]

    word_id = []

    if vocab is None:
        word_id = list(set.union(*vocab_embeddings))
    else:
        word_id = set(vocab)
        union = set.union(*vocab_embeddings)
        [
            print("Word " + str(w) + " not found in any embedding")
            for w in word_id - union
        ]
        word_id = list(word_id.intersection(union))

    print("The final embedding will have " + str(len(word_id)) + " words.")

    for i_voc, voc in enumerate(vocab_embeddings):
        print("Embedding " + str(i_voc) + " has " + str(len(voc)) + " words.")
        print(
            "We will generate "
            + str(len(set(word_id) - voc))
            + " words for the embedding "
            + str(i_voc)
        )

    print()

    printTrace("Building matrix for word generation...")
    generation_vocab_matrix = [
        [x for x in range(len(embeddings_path))] for x in range(len(embeddings_path))
    ]
    nn_vocab = [defaultdict() for x in range(len(embeddings_path))]

    for x, emb1 in enumerate(vocab_embeddings):
        vocab_to_generate = set(word_id) - emb1
        for y, emb2 in enumerate(vocab_embeddings):
            generation_vocab_matrix[y][x] = list(vocab_to_generate.intersection(emb2))
            vocab_to_generate = vocab_to_generate - emb2

    # print(generation_vocab_matrix)

    printTrace("===> Calculating nearest neighbors <===")

    for i_emb_path, emb_path in enumerate(embeddings_path):

        printTrace("Loading file: " + str(emb_path))
        emb = load_embedding(
            emb_path,
            vocabulary=None,
            lower=False,
            length_normalize=True,
            normalize_dimensionwise=False,
            delete_duplicates=True,
        )

        for i_g, g in enumerate(generation_vocab_matrix[i_emb_path]):
            if len(g) > 0:
                # print('G: ' + str(g))
                m = emb.words_to_matrix(g)  # generation_vocab_matrix[i_emb_path][i_g])
                m = emb.words_to_matrix(g)  # generation_vocab_matrix[i_emb_path][i_g])

                # print(len(m))
                # print(generation_vocab_matrix[x][gi])

                interset_vocab = list(
                    set.intersection(
                        vocab_embeddings[i_emb_path], vocab_embeddings[i_g]
                    )
                )

                M = emb.words_to_matrix(interset_vocab)

                total_words = len(m)

                for i_batch, mb in enumerate(batch(m, batch_size)):

                    string = (
                        "<"
                        + str(datetime.datetime.now())
                        + ">  "
                        + "Using Embedding "
                        + str(i_emb_path)
                        + " to generate vocab for Embedding "
                        + str(i_g)
                        + ":  "
                        + str(int(100 * (i_batch * batch_size) / total_words))
                        + "%"
                    )
                    print(string, end="\r")

                    # print(np.asarray(mb).shape)
                    # print(np.asarray(M).shape)

                    result = cosine_knn(mb, M, k)
                    for i_result, indexes in enumerate(result):
                        nn_vocab[i_g][g[i_result + (batch_size * i_batch)]] = [
                            interset_vocab[i] for i in indexes
                        ]

                print()

    printTrace("===> Calculating meta embedding <===")

    matrix = np.zeros([len(word_id), dims], dtype=float)
    total_words = len(word_id)

    for x, emb_path in enumerate(embeddings_path):
        printTrace("Loading file: " + str(emb_path))

        emb = load_embedding(
            emb_path,
            vocabulary=None,
            lower=False,
            length_normalize=False,
            normalize_dimensionwise=False,
            delete_duplicates=True,
        )

        for i, xb in enumerate(batch(word_id, batch_size)):

            string = (
                "<"
                + str(datetime.datetime.now())
                + ">  "
                + "Embedding "
                + str(x)
                + ": "
                + str(int(100 * (i * batch_size) / total_words))
                + "%"
            )
            print(string, end="\r")
            m = []
            for w in xb:

                try:
                    m.append(emb.word_to_vector(w))
                except KeyError as r:
                    try:
                        lw = nn_vocab[x][w]
                        v = np.zeros([dims], dtype=float)
                        for word in lw:
                            v += emb.word_to_vector(word)

                    except KeyError as r:
                        raise ValueError(
                            "Something went wrong in the word generation process"
                        )

                    m.append(normalize_vector(v / k))

            matrix[i * batch_size : i * batch_size + len(m)] = matrix_add(
                matrix[i * batch_size : i * batch_size + len(m)], m
            )
            # matrix[x*batch_size:x*batch_size+len(m)] = matrix_add(matrix[x*batch_size:x*batch_size+len(m)],m)
        print()

    printTrace("===> Printing meta embedding to file <===")
    with open(out_path, "w+") as file:
        print("%d %d" % (len(word_id), dims), file=file)
        for wi, w in enumerate(word_id):
            print(w + " " + " ".join(["%.6g" % x for x in matrix[wi]]), file=file)

            if wi % 1000 == 0:
                string = (
                    "<"
                    + str(datetime.datetime.now())
                    + "> "
                    + "Printing to file :"
                    + str(int(100 * wi / total_words))
                    + "%"
                )
                print(string, end="\r")

        print()


def average_embeddings(
    embeddings_path, out_path, vocab,
):

    dims = 0
    for p in embeddings_path:
        if dims:

            d = get_dimensions(p)
            if d is None:
                raise ValueError("The embeddings must be in the word2vec format")

            elif dims == get_dimensions(p):
                continue

            else:
                raise ValueError(
                    "All the embeddings must have the same number of dimensions"
                    " and the embeddings must be in the word2vec format"
                )
        else:
            dims = get_dimensions(p)
            if dims is None:
                raise ValueError("The embeddings must be in the word2vec format")

    vocab_embeddings = [vocab_from_path(x) for x in embeddings_path]

    if vocab is None:
        word_id = list(set.union(*vocab_embeddings))
    else:
        word_id = set(vocab)
        union = set.union(*vocab_embeddings)
        [
            print("Word " + str(w) + " not found in any embedding")
            for w in word_id - union
        ]
        word_id = list(word_id.intersection(union))

    print("The final embedding will have " + str(len(word_id)) + " words.")

    printTrace("===> Calculating meta embedding (no OOV) <===")

    matrix = np.zeros([len(word_id), dims], dtype=float)
    matrix_div = np.zeros([len(word_id)], dtype=float)

    total_words = len(word_id)

    for x, emb_path in enumerate(embeddings_path):
        printTrace("Loading file: " + str(emb_path))

        emb = load_embedding(
            emb_path,
            vocabulary=None,
            lower=False,
            length_normalize=False,
            normalize_dimensionwise=False,
            delete_duplicates=True,
        )

        for i_word, word in enumerate(word_id):
            if i_word % 1000 == 0:
                string = (
                    "<"
                    + str(datetime.datetime.now())
                    + ">  "
                    + "Embedding "
                    + str(x)
                    + ": "
                    + str(int(100 * (i_word / total_words)))
                    + "%"
                )
                print(string, end="\r")

            try:
                matrix[i_word] += emb.word_to_vector(word)
                matrix_div[i_word] += 1
            except KeyError as r:
                continue

    matrix = matrix / matrix_div[:, None]

    print()

    printTrace("===> Printing meta embedding to file <===")
    with open(out_path, "w+") as file:
        print("%d %d" % (len(word_id), dims), file=file)
        for wi, w in enumerate(word_id):
            print(w + " " + " ".join(["%.6g" % x for x in matrix[wi]]), file=file)

            if wi % 1000 == 0:
                string = (
                    "<"
                    + str(datetime.datetime.now())
                    + "> "
                    + "Printing to file :"
                    + str(int(100 * wi / total_words))
                    + "%"
                )
                print(string, end="\r")

        print()


if __name__ == "__main__":
    main()
