import argparse
import os
from utils import printTrace, get_dimensions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('-o', '--output_directory', type=str)
    parser.add_argument('-m', '--method', choices=['PCA', 'tSVD', 'DRA'], default='DRA')
    parser.add_argument('-n', '--reduce_to', type=int, default=300)
    parser.add_argument('-b', '--do_in_batches', action='store_true')

    args = parser.parse_args()

    files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if
             os.path.isfile(os.path.join(args.directory, f))]

    for i_file, file in enumerate(files):
        printTrace('Dimensionality reduction: Embedding ' + str(i_file) + ' of ' + str(len(files)) + ' : ' + str(file))

        excec_com = 'python3 dimensionality_reduction.py -i ' + str(file) + ' -m ' + str(args.method) + ' -o ' +\
                    args.output_directory + file.split('/')[-1] + '_' + str(args.method) +'.vec -n ' +\
                    str(args.reduce_to) + (' -b ' if args.do_in_batches else '')
        print(excec_com)
        os.system(excec_com)


if __name__ == '__main__':
    main()
