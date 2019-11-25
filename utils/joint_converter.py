import sys
sys.path.insert(0, '../')

from embedding import load_embedding
from utils import vocab_from_path


Joint_path = '../../Embeddings/'

print("====ENGLISH-SPANISH===")

words_eng = []
words_eng.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENES.emb'))
words_eng.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENIT.emb'))
english_words = list(set.intersection(*words_eng))

words_es = []
words_es.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENES.emb'))
words_es.append(vocab_from_path(Joint_path+'JOINTC-HYB-ESIT.emb'))
spanish_words = list(set.intersection(*words_es))


emb = load_embedding(Joint_path+'JOINTC-HYB-ENES.emb',  length_normalize=False, delete_duplicates=True)

with open('../../Embeddings/separated/JointENES.vec','w') as file:

    print(str(len(spanish_words) +len(english_words)) + ' 300', file=file)

    for word in english_words:
        print('en/'+word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)
    for word in spanish_words:
        print('es/'+word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)




print("====ENGLISH-BASQUE===")
words_eng = []
words_eng.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENES.emb'))
words_eng.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENEU.emb'))
english_words = list(set.intersection(*words_eng))

words_eu = []
words_eu.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENEU.emb'))
words_eu.append(vocab_from_path(Joint_path+'JOINTC-HYB-ESEU.emb'))
basque_words = list(set.intersection(*words_eu))


emb = load_embedding(Joint_path+'JOINTC-HYB-ENEU.emb',  length_normalize=False, delete_duplicates=True)

with open('../../Embeddings/separated/JointENEU.vec','w') as file:

    print(str(len(basque_words) + len(english_words)) + ' 300', file=file)

    for word in english_words:
        print('en/'+word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)
    for word in basque_words:
        print('eu/'+word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)


print("====SPANISH-BASQUE===")


words_es = []
words_es.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENES.emb'))
words_es.append(vocab_from_path(Joint_path+'JOINTC-HYB-ESEU.emb'))
spanish_words = list(set.intersection(*words_es))

words_eu = []
words_eu.append(vocab_from_path(Joint_path+'JOINTC-HYB-ENEU.emb'))
words_eu.append(vocab_from_path(Joint_path+'JOINTC-HYB-ESEU.emb'))
basque_words = list(set.intersection(*words_eu))


emb = load_embedding(Joint_path+'JOINTC-HYB-ESEU.emb',  length_normalize=False, delete_duplicates=True)

with open('../../Embeddings/separated/JointESEU.vec','w') as file:

    print(str(len(basque_words) + len(spanish_words)) + ' 300', file=file)

    for word in spanish_words:
        print('en/'+word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)
    for word in basque_words:
        print('eu/'+word + ' ' + ' '.join(['%.6g' % x for x in emb.word_to_vector(word)]), file=file)

from embedding import load_embedding