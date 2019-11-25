import numpy as np
from tensorflow_functions import matrix_analogy
import argparse
from embedding import load_embedding
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embedding', required=True)
    parser.add_argument('-n', '--name_of_embedding', default=None)
    parser.add_argument('-l', '--lowercase_dataset', action='store_true')
    parser.add_argument('-d', '--dataset_path', default='AnalogyDataset/questions-words.txt')
    parser.add_argument('-b', '--batch_size', type=int, default=512)

    args = parser.parse_args()

    print(' >>> loading embedding <<< ')
    emb = load_embedding(args.embedding, lower=False, length_normalize=True, delete_duplicates=True)
    name = args.embedding if args.name_of_embedding is None else args.name_of_embedding

    if not os.path.exists('Results'):
        os.makedirs('Results')

    print('>>> Results deleting oov <<< ')

    results = evaluate_analogy(emb.vocabulary.word_id, emb.vectors, dataset_path=args.dataset_path, lowercase=args.lowercase_dataset, BATCH_SIZE=args.batch_size)
    print_to_csv_analogy(results, name=name, filenameResults='Results/Analogy_Results_delete.csv')

    print()
    print()
    print('>>> Result using mean of all word vectors as OOV <<<')

    #emb.vocabulary.word_id['<<UKN>>'] = len(emb.words)
    emb.vectors = np.append(emb.vectors, [np.mean(emb.vectors, axis=0)], axis=0)

    results = evaluate_analogy(emb.vocabulary.word_id, emb.vectors, dataset_path=args.dataset_path, lowercase=args.lowercase_dataset, BATCH_SIZE=args.batch_size, backoff_vector=len(emb.vectors)-1)
    print_to_csv_analogy(results, name=name, filenameResults='Results/Analogy_Results_mean.csv')

    print('Results have been exported in csv format to the Results folder')


def evaluate_analogy(word2ind,matrix,dataset_path, lowercase, BATCH_SIZE ,backoff_vector=None):

    output_csv = ''

    f = open(dataset_path, errors='surrogateescape')
    categories = []
    src1 = []
    trg1 = []
    src2 = []
    trg2 = []
    for line in f:
        if line.startswith(': '):
            name = line[2:-1]
            is_syntactic = name.startswith('gram')
            categories.append({'name': name, 'is_syntactic': is_syntactic, 'total': 0, 'oov': 0})
        else:
            try:
                ind = [word2ind[word.lower() if lowercase else word] for word in line.split()]
                src1.append(ind[0])
                trg1.append(ind[1])
                src2.append(ind[2])
                trg2.append(ind[3])
                categories[-1]['total'] += 1
            except KeyError:

                if backoff_vector is not None:
                    l = line.split()
                    try:
                        src1.append(word2ind[l[0].lower() if lowercase else l[0]])
                    except KeyError:
                        src1.append(backoff_vector)
                    try:
                        trg1.append(word2ind[l[1].lower() if lowercase else l[1]])
                    except KeyError:
                        trg1.append(backoff_vector)
                    try:
                        src2.append(word2ind[l[2].lower() if lowercase else l[2]])
                    except KeyError:
                        src2.append(backoff_vector)
                    try:
                        trg2.append(word2ind[l[3].lower() if lowercase else l[3]])
                    except KeyError:
                        trg2.append(backoff_vector)

                    categories[-1]['total'] += 1
                else:
                    categories[-1]['oov'] += 1

    total = len(src1)
    print('>>> Calculating Analogy <<<')
    # Compute nearest neighbors using efficient matrix multiplication
    nn = []
    for i in range(0, total, BATCH_SIZE):

        string = 'Calculating analogies: ' + str(int(100*i/total)) +'%'
        print(string, end="\r")

        j = min(i + BATCH_SIZE, total)


        similarities = []

        try: #Try GPU, if OOM use CPU
            similarities = matrix_analogy(matrix[src1[i:j]], matrix[trg1[i:j]], matrix[src2[i:j]], matrix)
        except:
            similarities = (matrix[src2[i:j]] - matrix[src1[i:j]] + matrix[trg1[i:j]]).dot(matrix.T)
        #if len(matrix) > 1000000:


        #else:


        similarities[range(j - i), src1[i:j]] = -1
        similarities[range(j - i), trg1[i:j]] = -1
        similarities[range(j - i), src2[i:j]] = -1
        nn += np.argmax(similarities, axis=1).tolist()
    nn = np.array(nn)

    # Compute and print accuracies
    semantic = {'correct': 0, 'total': 0, 'oov': 0}
    syntactic = {'correct': 0, 'total': 0, 'oov': 0}
    ind = 0
    for category in categories:
        current = syntactic if category['is_syntactic'] else semantic
        correct = np.sum(nn[ind:ind + category['total']] == trg2[ind:ind + category['total']])
        current['correct'] += correct
        current['total'] += category['total']
        current['oov'] += category['oov']
        ind += category['total']

        print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} | {2}'.format(
                    category['total'] / (category['total'] + category['oov']),
                    correct / category['total'],
                    category['name']))

        output_csv = output_csv + ', ' + str(correct / category['total'])

    print('-' * 80)
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%} (sem:{2:7.2%}, syn:{3:7.2%})'.format(
        (semantic['total'] + syntactic['total']) / (
                    semantic['total'] + syntactic['total'] + semantic['oov'] + syntactic['oov']),
        (semantic['correct'] + syntactic['correct']) / (semantic['total'] + syntactic['total']),
        semantic['correct'] / semantic['total'],
        syntactic['correct'] / syntactic['total']))

    output_csv = output_csv + ', ' + str((semantic['total'] + syntactic['total']) / (
                    semantic['total'] + syntactic['total'] + semantic['oov'] + syntactic['oov'])) + ', ' \
                 + str((semantic['correct'] + syntactic['correct']) / (semantic['total'] + syntactic['total'])) + ', '\
                 + str(semantic['correct'] / semantic['total']) + ', '\
                 + str(syntactic['correct'] / syntactic['total'])

    return output_csv


def print_to_csv_analogy(txtResults, name=None, filenameResults='Results.csv'):
    if name:
        txtResults = str(name) + "," + txtResults

    with open(filenameResults, 'a+') as file:
        print('%s' % (str(txtResults)), file=file)



if __name__ == '__main__':
    main()



