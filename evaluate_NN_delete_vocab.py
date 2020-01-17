import numpy as np
import datetime
import argparse
from collections import defaultdict
import sys
from tensorflow_functions import matrix_add, cosine_knn_batches, cosine_knn
from embedding import load_embedding, Embedding
from utils import get_dimensions
from utils import vocab_from_path, normalize_vector, printTrace, batch
from similarity_datasets import *
from evaluate_similarity import similarity_emd
from vocabulary import Vocabulary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--embedding", type=str, required=True)
    parser.add_argument("-c", "--emb_4_generation", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=1024)
    parser.add_argument("-k", "--num_nearest_neighbor", type=int, default=10)

    args = parser.parse_args()

    dims = get_dimensions(args.embedding)

    if dims != get_dimensions(args.emb_4_generation):
        raise ValueError(
            "All the embeddings must have the same number of dimensions and the embeddings must be in the word2vec format"
        )

    printTrace("Reading vocab...")

    vocab_emb = vocab_from_path(args.embedding)
    vocab_cross = vocab_from_path(args.emb_4_generation)
    dataset = get_dataset(args.dataset)
    vocab_to_generate = list(set(np.append((dataset.X[:, 0]), (dataset.X[:, 1]))))
    vocab_to_generate_set = set(vocab_to_generate)
    vocab_emb_delete = [x for x in vocab_emb if x not in vocab_to_generate_set]

    total_vocab = set.union(set(vocab_emb_delete), set(vocab_cross))
    interset_vocab = list(set.intersection(set(vocab_emb_delete), set(vocab_cross)))


    print("Final embedding will have " + str(len(total_vocab)) + " words")
    print("We will generate " + str(len(vocab_to_generate)) + " words")

    emb = load_embedding(
        args.emb_4_generation,
        vocabulary=None,
        lower=False,
        length_normalize=True,
        normalize_dimensionwise=False,
        delete_duplicates=True,
    )

    m = emb.words_to_matrix(vocab_to_generate)
    M = emb.words_to_matrix(interset_vocab)

    nn = []

    for i_batch, mb in enumerate(batch(m, args.batch_size)):

        string = (
            "<"
            + str(datetime.datetime.now())
            + ">  "
            + "Using Embedding "
            + str(args.emb_4_generation)
            + " to generate vocab for Embedding "
            + str(args.embedding)
            + ":  "
            + str(int(100 * (i_batch * args.batch_size) / len(m)))
            + "%"
        )
        print(string, end="\r")

        # print(np.asarray(mb).shape)
        # print(np.asarray(M).shape)

        result = cosine_knn(mb, M, args.num_nearest_neighbor)

        for i_result, indexes in enumerate(result):
            nn.append([interset_vocab[i] for i in indexes])

    del emb

    printTrace("===> Generating new_vocab <===")

    emb = load_embedding(
        args.embedding,
        vocabulary=vocab_emb_delete,
        lower=False,
        length_normalize=False,
        normalize_dimensionwise=False,
        delete_duplicates=True,
    )

    new_vectors = []
    for i_word, word in enumerate(vocab_to_generate):
        if i_word % 1000 == 0:
            string = (
                "<"
                + str(datetime.datetime.now())
                + ">  "
                + "Generating vocab "
                + ": "
                + str(int(100 * i_word / len(vocab_to_generate)))
                + "%"
            )
            print(string, end="\r")

        try:
            lw = nn[i_word]
            v = np.zeros([dims], dtype=float)
            for word_nn in lw:
                v += emb.word_to_vector(word_nn)

        except KeyError as r:
            raise ValueError("Something went wrong in the word generation process")

        new_vectors.append(v / args.num_nearest_neighbor)

    print()

    del emb

    printTrace("===> Loading embeddings to compare <===")
    emb_generated = Embedding(vocabulary=Vocabulary(vocab_to_generate), vectors=new_vectors)
    emb_original = load_embedding(
        args.embedding,
        vocabulary=vocab_to_generate,
        lower=False,
        length_normalize=False,
        normalize_dimensionwise=False,
        delete_duplicates=True,
    )

    printTrace("===> Evaluate <===")

    print("Original Embedding: ", end="")
    print(
        similarity_emd(
            emb_original,
            dataset.X,
            dataset.y,
            backoff_vector=None,
            lower=False,
            lang1prefix=None,
            lang2prefix=None,
        )
    )
    print("Generated Embedding: ", end="")
    print(
        similarity_emd(
            emb_generated,
            dataset.X,
            dataset.y,
            backoff_vector=None,
            lower=False,
            lang1prefix=None,
            lang2prefix=None,
        )
    )


if __name__ == "__main__":
    main()
