import argparse
import os
from embedding import load_embedding
from utils import printTrace, vocab_from_path


def main():
    parser = argparse.ArgumentParser()
    inputtype = parser.add_mutually_exclusive_group(required=True)
    inputtype.add_argument('-i', '--embedding', type=str)
    inputtype.add_argument('-d', '--directory', type=str)
    parser.add_argument('-o', '--output', required=True)

    parser.add_argument('-v', '--vocab', default=None)
    parser.add_argument('-nl', '--length_normalize', action='store_true')
    parser.add_argument('-nd', '--normalize_dimensionwise', action='store_true')
    parser.add_argument('-l', '--lower', action='store_true')

    outputtype = parser.add_mutually_exclusive_group(required=True)
    outputtype.add_argument('-w2v', '--word2vec', action='store_true')
    outputtype.add_argument('-glv', '--glove', action='store_true')
    args = parser.parse_args()

    if args.embedding:
        emb_converter(args.embedding, args.output, args)
    else:
        files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if
                 os.path.isfile(os.path.join(args.directory, f))]

        for i_file, file in enumerate(files):
            printTrace('Converting Embedding ' + str(i_file) + ' of ' + str(len(files)) + ' : ' + str(file))
            emb_converter(file, args.output+'/'+file.split('/')[-1], args)


def emb_converter(path_input, path_output, args):
    printTrace('Loading Embedding ' + str(path_input) + '...')
    format = 'bin' if path_input.split('/')[-1].split('.')[-1] == 'bin' else 'text'

    emb = load_embedding(path_input, format=format,
                             vocabulary=None if args.vocab is None else vocab_from_path(args.vocab),
                             length_normalize=args.length_normalize,
                             normalize_dimensionwise=args.normalize_dimensionwise, to_unicode=True,
                             lower=args.lower, path2='', delete_duplicates=True, method_vgg="delete")

    printTrace('Saving result to ' + str(path_output) + '...')
    emb.export(path=path_output, printHeader=args.word2vec)

    printTrace('Done.')

if __name__ == '__main__':
    main()
