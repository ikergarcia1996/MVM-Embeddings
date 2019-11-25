import sys
import argparse
sys.path.insert(0, '../')
from embedding import load_embedding

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
args = parser.parse_args()

emb = load_embedding(args.input,  length_normalize=False, delete_duplicates=True)

with open('/home/iker/Documents/Embeddings/Separated/'+args.input.split('/')[-1],'w') as file:
    print(str(len(emb.words)) + ' ' + str(emb.dims), file=file)

    for word in emb.words:
        print(''.join(word.split('/')[1:]) + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)



