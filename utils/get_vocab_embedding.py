import sys
import argparse
sys.path.insert(0, '../')

from utils import vocab_from_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embedding', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    vocab = vocab_from_path(args.embedding)

    with open(args.output,'w+') as file:
        for word in vocab:
            print(word, file=file)

    print('Done.')


if __name__ == '__main__':
    main()


