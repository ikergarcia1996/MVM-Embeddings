import argparse
import os
import sys
import os
sys.path.insert(0, '../')
from embedding import load_embedding

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-p', '--prefix', required=True)
args = parser.parse_args()

emb = load_embedding(args.input,
                     vocabulary=None,
                     length_normalize=False,
                     normalize_dimensionwise=False, to_unicode=True,
                     lower=False, delete_duplicates=True)

n_words = 0

with open(args.output,'w') as file:
    for word in emb.words:
        if word.split('/')[0] == args.prefix:
            print(''.join(word.split('/')[1:]) + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)
            n_words+=1

excec_com = 'sed -i \'1s/^/' + str(n_words) + ' ' + str(emb.dims) + '\\n/\' ' + str(args.output)
print(excec_com)
os.system(excec_com)