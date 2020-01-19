import numpy as np
from collections import defaultdict
from tensorflow_functions import cosine_knn
from embedding import load_embedding
from utils import vocab_from_path, normalize_vector, printTrace, batch
import datetime
import argparse
import os
import shutil


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
        concatenate_embeddings_generate(
            embeddings_path=args.embeddings,
            out_path=args.output,
            vocab=vocab_from_path(args.vocabulary) if args.vocabulary else None,
            batch_size=args.batch_size,
            k=args.num_nearest_neighbor,
        )
    else:
        concatenate_embeddings(
            embeddings_path=args.embeddings,
            out_path=args.output,
            vocab=vocab_from_path(args.vocabulary) if args.vocabulary else None,
        )


def concatenate_embeddings_generate(
    embeddings_path, out_path, vocab=None, batch_size=1024, k=10
):
    printTrace("Reading vocab...")

    # [[vocab_emb1], [vocab_emb_2], ...]
    vocab_embeddings = [vocab_from_path(x) for x in embeddings_path]

    word_id = set()

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

    printTrace("===> Calculating nearest neighbors <===")

    for i_emb_path, emb_path in enumerate(embeddings_path):

        printTrace("Loading file: " + str(emb_path))
        emb = load_embedding(
            emb_path,
            vocabulary=None,
            length_normalize=True,
            normalize_dimensionwise=False,
            delete_duplicates=True,
        )

        for i_g, g in enumerate(generation_vocab_matrix[i_emb_path]):
            if len(g) > 0:
                # print('G: ' + str(g))
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

                    result = cosine_knn(mb, M, k)
                    for i_result, indexes in enumerate(result):
                        nn_vocab[i_g][g[i_result + (batch_size * i_batch)]] = [
                            interset_vocab[i] for i in indexes
                        ]

                print()

    printTrace("===> Calculating meta embedding <===")

    total_words = len(word_id)
    first_emb = True

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    total_dims = 0

    for x, emb_path in enumerate(embeddings_path):
        matrix = []
        printTrace("Loading file: " + str(emb_path))

        emb = load_embedding(
            emb_path,
            vocabulary=None,
            length_normalize=True,
            normalize_dimensionwise=False,
            delete_duplicates=True,
        )

        total_dims += emb.dims

        string = "<" + str(datetime.datetime.now()) + ">  " + "Embedding " + str(x)
        print(string, end="\r")

        actual_matrix = []

        for wi, w in enumerate(word_id):
            m = np.zeros([emb.dims], dtype=float)
            try:
                m = emb.word_to_vector(w)
            except KeyError as r:
                try:
                    lw = nn_vocab[x][w]
                    v = np.zeros([emb.dims], dtype=float)
                    for word in lw:
                        v += emb.word_to_vector(word)

                except KeyError as r:
                    raise ValueError(
                        "Something went wrong in the word generation process"
                    )

                m = normalize_vector(v / k)

            matrix.append(m)

            if wi % 1000 == 0:
                string = (
                    "<"
                    + str(datetime.datetime.now())
                    + "> "
                    + "Calculating meta embeddind for embedding "
                    + str(x)
                    + ": "
                    + str(int(100 * wi / total_words))
                    + "%"
                )
                print(string, end="\r")
        print()

        with open("tmp/" + str(x), "w") as file:
            for wi, w in enumerate(word_id):
                if first_emb:
                    print(
                        w + " " + " ".join(["%.6g" % x for x in matrix[wi]]), file=file
                    )
                else:
                    print(" ".join(["%.6g" % x for x in matrix[wi]]), file=file)

                if wi % 1000 == 0:
                    string = (
                        "<"
                        + str(datetime.datetime.now())
                        + "> "
                        + "Saving embedding "
                        + str(x)
                        + " to file : "
                        + str(int(100 * wi / total_words))
                        + "%"
                    )
                    print(string, end="\r")

            print()

        first_emb = False

    printTrace("Concatenation...")

    excec_com = "paste -d ' ' "
    for x in range(len(embeddings_path)):
        excec_com = excec_com + "tmp/" + str(x) + " "
    excec_com = excec_com + "> " + str(out_path)
    print(excec_com)
    os.system(excec_com)

    excec_com = (
        "sed -i '1s/^/"
        + str(len(word_id))
        + " "
        + str(total_dims)
        + "\\n/' "
        + str(out_path)
    )
    print(excec_com)
    os.system(excec_com)

    try:
        os.system("rm -rf tmp")
    except:
        print("Could not delete the tmp folder, do it manually")

    printTrace("Done. Meta embedding saved in " + str(out_path))


def concatenate_embeddings(
    embeddings_path, out_path, vocab,
):
    printTrace("===> Calculating meta embedding (No OOV) <===")

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

    first_emb = True

    if not os.path.exists("tmp_conc"):
        os.makedirs("tmp_conc")

    total_dims = 0

    for x, emb_path in enumerate(embeddings_path):
        matrix = []
        printTrace("Loading file: " + str(emb_path))

        emb = load_embedding(
            emb_path,
            vocabulary=None,
            length_normalize=True,
            normalize_dimensionwise=False,
            delete_duplicates=True,
        )

        total_dims += emb.dims

        string = "<" + str(datetime.datetime.now()) + ">  " + "Embedding " + str(x)
        print(string, end="\r")

        for wi, w in enumerate(word_id):
            m = np.zeros([emb.dims], dtype=float)
            try:
                m = emb.word_to_vector(w)
            except KeyError as r:
                pass

            matrix.append(m)

            if wi % 1000 == 0:
                string = (
                    "<"
                    + str(datetime.datetime.now())
                    + "> "
                    + "Calculating meta embeddind for embedding "
                    + str(x)
                    + ": "
                    + str(int(100 * wi / len(word_id)))
                    + "%"
                )
                print(string, end="\r")
        print()

        with open("tmp_conc/" + str(x), "w+", encoding="utf-8") as file:
            for wi, w in enumerate(word_id):
                if first_emb:
                    print(
                        w + " " + " ".join(["%.6g" % x for x in matrix[wi]]), file=file
                    )
                else:
                    print(" ".join(["%.6g" % x for x in matrix[wi]]), file=file)

                if wi % 1000 == 0:
                    string = (
                        "<"
                        + str(datetime.datetime.now())
                        + "> "
                        + "Saving embedding "
                        + str(x)
                        + " to file : "
                        + str(int(100 * wi / len(word_id)))
                        + "%"
                    )
                    print(string, end="\r")

            print()

        first_emb = False

    printTrace("Concatenation...")

    excec_com = "paste -d ' ' "
    for x in range(len(embeddings_path)):
        excec_com = excec_com + "tmp_conc/" + str(x) + " "
    excec_com = excec_com + "> " + str(out_path)
    print(excec_com)
    os.system(excec_com)

    excec_com = (
        "sed -i '1s/^/"
        + str(len(word_id))
        + " "
        + str(total_dims)
        + "\\n/' "
        + str(out_path)
    )
    print(excec_com)
    os.system(excec_com)

    try:
        shutil.rmtree("/tmp_conc")
    except:
        print("Could not delete the tmp folder, do it manually")

    printTrace("Done. Meta embedding saved in " + str(out_path))


if __name__ == "__main__":
    main()
