import sys
import argparse
sys.path.insert(0, '../')
from utils import vocab_from_path, printTrace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embeddings', nargs='+', required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    printTrace('Loading vocabulary from embeddings...')
    vocab_embeddings = [vocab_from_path(x) for x in args.embeddings]
    union_vocab = (set.union(*vocab_embeddings))
    printTrace('Te union of the vocabulary has ' + str(len(union_vocab)) + ' words.')
    printTrace('Printing vocabulary in ' + args.output + '...')
    with open(args.output, 'w+') as file:
        for word in union_vocab:
            print(word, file=file)


if __name__ == '__main__':
    main()
