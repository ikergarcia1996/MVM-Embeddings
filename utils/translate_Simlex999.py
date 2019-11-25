from py_translator import Translator
import pandas as pd
import os
import sys
sys.path.insert(0, '../')
from utils import batch

dataset_path = '../SimilarityDataset/ENES-SIM999.txt'
output_path = '../SimilarityDataset/ES-SIM999.txt'
validation_file_path = '../SimilarityDataset/SimTranslate_validation.txt'
batch_size = 100


def main():
    global dataset_path
    global output_path
    global validation_file_path

    if os.path.exists(output_path):
        os.system('rm '+ output_path)
        print('Removed ' + output_path)
    if os.path.exists(validation_file_path):
        os.system('rm ' + validation_file_path)
        print('Removed ' + validation_file_path)


    data = pd.read_csv(dataset_path, sep="\t", header=None).values

    # If file exits restore progress
    # if os.path.isfile(output_path):
    #     restore = pd.read_csv(dataset_path, sep="\t", header=None)
    #     l = len(restore)
    #     print('Restoring progress. Line ' + str(l))
    #     pairs = pairs[l:]
    #     golds = golds[l:]

    with open(validation_file_path, 'w+') as validfile:
        print('Origninal \t Translation', file=validfile)



    for slice in batch(data, batch_size):
        text = ''
        es_list = []
        en_list = []
        gold_list = []

        for en, es, gold in slice:
            es_list.append(es)
            en_list.append(en)
            gold_list.append(gold)
            text = text + en + '. '

        translated = Translator().translate(text=text, dest='es').text
        en_list_translated = translated.split('.')

        assert (len(en_list_translated)-1 == len(en_list))
        with open(validation_file_path, 'a+') as validfile:
            with open(output_path, 'a+') as output_file:
                for i in range(len(en_list)):
                    print(en_list[i] + '\t' + en_list_translated[i].strip().lower(), file=validfile)
                    print(en_list_translated[i].strip().lower() + '\t' + es_list[i] + '\t' + str(gold_list[i]), file=output_file)


    # for i, pair in enumerate(pairs):
    #    en, es = pair
    #    t = Translator().translate(text=en, dest='es').text
    #    with open(validation_file_path, 'a+') as validfile:
    #        print(en + '\t' + t, file=validfile)

    #   with open(output_path, 'a+') as output_path:
    #        print(t + '\t' + es + '\t' + golds[i], file=output_path)

    print('Done')


if __name__ == '__main__':
    main()









