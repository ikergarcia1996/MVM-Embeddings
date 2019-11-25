import sys
import argparse
sys.path.insert(0, '../')
from embedding import load_embedding

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--embeddings', nargs='+', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

embs = []
total_words = 0
dims = 0
for emb in args.embeddings:
    embs.append(load_embedding(emb,  length_normalize=False, delete_duplicates=True))

for emb in embs:
    total_words += len(emb.words)

dims = embs[0].dims

for emb in embs:
    if emb.dims!=dims:
        raise ValueError('All the embeddings must have the same number of dimensions and the embeddings must be in the word2vec format')


with open(args.output, 'w+', encoding='utf-8') as file:
    print(str(total_words) + ' ' + str(dims), file=file)

    for emb in embs:
        for word in emb.words:
            print(word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)




