import scipy.stats
from six import iteritems
from similarity_datasets import get_datasets
from embedding import load_embedding
import argparse
import logging
import numpy as np
import os
from utils import printTrace, vocab_from_path


def main():
    parser = argparse.ArgumentParser()
    inputtype = parser.add_mutually_exclusive_group(required=True)
    inputtype.add_argument('-i', '--embedding', type=str)
    inputtype.add_argument('-d', '--directory', type=str)

    #parser.add_argument('-n', '--name_of_embedding', default=None)
    parser.add_argument('-l', '--lowercase_dataset', action='store_true')
    parser.add_argument('-lg', '--language', nargs='+', default=['en'])

    parser.add_argument('-p', '--add_lang_prefix', action='store_true')

    parser.add_argument('-v', '--vocab', type=str, default=None)

    args = parser.parse_args()





    emb_list = []

    if args.embedding is not None:
        emb_list.append(args.embedding)
    else:
        emb_list = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if
                 os.path.isfile(os.path.join(args.directory, f))]



    for emb_i, emb_path in enumerate(emb_list):

        printTrace('Evaluating Embedding ' + str(emb_i+1) + ' of ' + str(len(emb_list)) + ' : ' + str(emb_path))

        emb = load_embedding(emb_path, vocabulary= (None if args.vocab is None else vocab_from_path(args.vocab)), lower=False, length_normalize=False, delete_duplicates=True)

        for lang in args.language:

            lang1prefix = None
            lang2prefix = None

            if args.add_lang_prefix:
                if lang == 'en':
                    lang1prefix = 'en'
                    lang2prefix = 'en'
                elif lang == 'es':
                    lang1prefix = 'es'
                    lang2prefix = 'es'
                elif lang == 'enes':
                    lang1prefix = 'en'
                    lang2prefix = 'es'
                else:
                    logging.warning('Language not supported, could not add prefix')

            if not os.path.exists('Results_' + lang):
                os.makedirs('Results_' + lang)

            print('>>> Results deleting oov <<< ')

            a, b = results_to_csv(evaluate_on_all(emb, backoff_vector=None, lowercase_dataset=args.lowercase_dataset, lang=lang, lang1prefix=lang1prefix, lang2prefix=lang2prefix), printRes=False, returnRes=True)
            export_to_csv(txtResults=a, txtCov=b, name=emb_path, filenameResults='Results_' + lang + '/Sim_Results_delete.csv', filenameCoverage='Results_' + lang + '/Sim_Coverage.csv')

            print('>>> Result using mean of all word vectors as OOV <<<')

            a, b = results_to_csv(evaluate_on_all(emb, backoff_vector=np.mean(emb.vectors, axis=0), lowercase_dataset=args.lowercase_dataset, lang=lang, lang1prefix=lang1prefix, lang2prefix=lang2prefix), printRes=False, returnRes=True)
            export_to_csv(txtResults=a, txtCov=b, name=emb_path, filenameResults='Results_' + lang + '/Sim_Results_mean.csv', filenameCoverage='Results_' + lang + '/Sim_Coverage.csv')

    print('Results have been exported in csv format to the Results folder')




def calculate_cosine_simil(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# [ES] Dado un embedding y dos palabras (string), devulve la similitud coseno entre ambas palabras, en caso de que alguna de las dos no exista en el embeddig se devolverá None
def similarity_2_words(e, word1, word2, lower=False):
    try:
        v1 = e.word_to_vector(word1, lower)
    except KeyError as err:
        logging.critical("The word {} does not exits in the embeding".format(word1))
        return None
    try:
        v2 = e.word_to_vector(word2, lower)
    except KeyError as err:
        logging.critical("The word {} does not exits in the embeding".format(word2))
        return None

    return calculate_cosine_simil(v1, v2)


def similarity_emd(embedding, X, gold, backoff_vector=None, lower=False, lang1prefix=None, lang2prefix=None):
    results = []
    gold_scores = []
    oov = 0
    for gold_score in range(len(gold)):

        w1 = None
        w2 = None
        missing = False

        try:
            if lang1prefix is None:
                w1 = embedding.word_to_vector(X[gold_score][0], lower)
            else:
                #print('lang1prefix +'/' + X[gold_score][0]'.lower())
                w1 = embedding.word_to_vector(lang1prefix +'/' + X[gold_score][0], lower)
        except KeyError as err:
            #print(X[gold_score][0])
            missing = True
            if backoff_vector is not None:
                w1 = backoff_vector

        try:
            if lang2prefix is None:
                w2 = embedding.word_to_vector(X[gold_score][1], lower)
            else:
                w2 = embedding.word_to_vector(lang2prefix + '/' + X[gold_score][1], lower)
        except KeyError as err:
            #print(X[gold_score][1])
            missing = True
            if backoff_vector is not None:
                w2 = backoff_vector

        if missing:
            oov += 1

        if w1 is not None and w2 is not None:
            cos = calculate_cosine_simil(w1, w2)
            results.append(cos)
            gold_scores.append(gold[gold_score])

    coverage = len(results) / (len(results) + oov)
    pearson = scipy.stats.pearsonr(gold_scores, results)[0]
    spearman = scipy.stats.spearmanr(gold_scores, results)[0]

    return{'coverage':coverage, 'pearson':pearson, 'spearman':spearman}

def evaluate_on_all(emb, backoff_vector=None, lowercase_dataset=False,lang='en', lang1prefix=None, lang2prefix=None):
    result = []

    for name, data in iteritems(get_datasets(lang)):
        d = {'dataset': name}
        d.update(similarity_emd(emb, data.X, data.y, backoff_vector, lowercase_dataset, lang1prefix=lang1prefix, lang2prefix=lang2prefix))
        result = np.append(result, d)
        print(d)


    return result


def results_to_csv(res, correlation='spearman', printRes=True, returnRes=False):
    assert correlation in ["spearman", "pearson"], "Unrecognized Correlation method"
    txtRest = ''
    txtCov = ''
    for y in res:
        txtRest = txtRest + str(y[correlation]) + ','
        txtCov = txtCov + str(y['coverage']) + ','


    if printRes:
        print(txtRest)
        print(txtCov)
    if returnRes:
        return txtRest, txtCov


def export_to_csv(txtResults, txtCov, name=None, filenameResults='Results.csv', filenameCoverage='Coverage.csv'):
    if name:
        txtResults = str(name) + "," + txtResults
        txtCov = str(name) + "," + txtCov

    with open(filenameResults, 'a+') as file:
        print('%s' % (str(txtResults)), file=file)

    with open(filenameCoverage, 'a+') as file:
        print('%s' % (str(txtCov)), file=file)


if __name__ == '__main__':
    main()
