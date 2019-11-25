import os
import subprocess
from utils import has_header
import sys
import argparse
import numpy as np

import random,copy,string
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Convolution1D, MaxPooling1D, Flatten
from tensorflow.python.keras.layers import Lambda, multiply, concatenate, Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from embedding import load_embedding


ADD_PREFIX = False
TEXT_EVAL = '[Train, Valid, Test, track3es-es, track4aes-en, track4bes-en, track5en-en]='


class Embedder(object):
    def __init__(self, dictname, wordvectdim):

        print('Loading ' + dictname + '...(This might take one or two minutes.)')
        self.wordtoindex   = dict()
        self.indextovector = []
        self.indextovector.append(np.zeros(wordvectdim))

        emb = load_embedding(dictname, length_normalize=False, to_unicode=False, lower=False, delete_duplicates=True)

        for i_word, word in enumerate(emb.words):
            self.wordtoindex.update([(word, i_word+1)])
            self.indextovector.append(emb.word_to_vector(word))

        del emb
        #print(self.indextovector.shape)
        self.indextovector = np.array(self.indextovector, dtype='float32')


    def _worker(self,args):
        wordtoindex   = dict()
        indextovector = []
        for line in args[0]:
            elements = line.split(' ')
            wordtoindex[elements[0]] = len(indextovector)+args[1]+1
            indextovector.append(np.array(elements[1:]).astype(float))
        return (wordtoindex,indextovector)
    def matrixize(self, sentencelist, sentencepad):
        indexlist = []
        for sentence in sentencelist:
            indexes = []
            for word in sentence:
                word = word.lower()
                if word not in self.wordtoindex: indexes.append(1)
                else: indexes.append(self.wordtoindex[word])
            indexlist.append(indexes)
        return self.indextovector[(pad_sequences(indexlist, maxlen=sentencepad, truncating='post', padding='post'))]

class STSTask():
    def __init__(self, c):
        self.c = c
    def load_resc(self,dictname):
        self.embed = Embedder(dictname, self.c['wordvectdim'])
    def load_data(self, trainfile, validfile, testfile, track3, track4a, track4b, track5):
        self.traindata= self._load_data(trainfile)
        self.validdata= self._load_data(validfile)
        self.testdata = self._load_data(testfile)
        self.track3 = self._load_data(track3)
        self.track4a = self._load_data(track4a)
        self.track4b = self._load_data(track4b)
        self.track5 = self._load_data(track5)
    def _load_data(self, filename):
        global ADD_PREFIX

        MULTILINGUAL = False

        prefix1=None
        prefix2=None
        print(filename)
        if filename=='STSdataset/sts-dev.csv' or filename=='STSdataset/sts-test.csv' or filename=='STSdataset/sts-train.csv' or filename=='STSdataset/track5en-en.csv':
            prefix1 = 'en/'
            prefix2 = 'en/'
        elif filename=='STSdataset/sts-train-es-split.csv' or filename == 'STSdataset/sts-dev-es-split.csv' or filename=='STSdataset/track3es-es.csv':
            prefix1 = 'es/'
            prefix2= 'es/'
        elif filename =='STSdataset/track4aes-en.csv':
            prefix1 = 'es/'
            prefix2= 'en/'
        elif filename=='STSdataset/track4bes-en.csv':
            prefix1 = 'en/'
            prefix2= 'es/'

        elif filename == 'STSdataset/sts-train-enes.csv' or filename == 'STSdataset/sts-dev-enes.csv' or filename=='STSdataset/sts-test-enes.csv':
            MULTILINGUAL = True


        print(prefix1)
        print(prefix2)

        s0,s1,labels = [],[],[]
        lines=open(filename,'r', encoding='utf8').read().splitlines()
        for line in lines:
            _,m,_,_, label, s0x, s1x = line.rstrip().split('\t')[:7]
            labels.append(float(label))

            if ADD_PREFIX:
                if MULTILINGUAL:
                    if m == 'NA':
                        prefix1 = 'es/'
                        prefix2 = 'es/'
                    else:
                        prefix1 = 'en/'
                        prefix2 = 'en/'

                s0.append([prefix1+word.lower() for word in word_tokenize(s0x) if word not in string.punctuation])
                s1.append([prefix2+word.lower() for word in word_tokenize(s1x) if word not in string.punctuation])

            else:

                s0.append([word.lower() for word in word_tokenize(s0x) if word not in string.punctuation])
                s1.append([word.lower() for word in word_tokenize(s1x) if word not in string.punctuation])


        m0 = self.embed.matrixize(s0, self.c['sentencepad'])
        m1 = self.embed.matrixize(s1, self.c['sentencepad'])
        classes = np.zeros((len(labels), self.c['num_classes']))
        for i, label in enumerate(labels):
            if np.floor(label) + 1 < self.c['num_classes']:
                classes[i, int(np.floor(label)) + 1] = label - np.floor(label)
            classes[i, int(np.floor(label))] = np.floor(label) - label + 1
        return {'labels': labels, 's0': s0, 's1': s1, 'classes': classes, 'm0': m0, 'm1': m1}

    def create_model(self):
        K.clear_session()
        input0 = Input(shape=(self.c['sentencepad'], self.c['wordvectdim']))
        input1 = Input(shape=(self.c['sentencepad'], self.c['wordvectdim']))
        Convolt_Layer=[]
        MaxPool_Layer=[]
        Flatten_Layer=[]
        for kernel_size, filters in self.c['cnnfilters'].items():
            Convolt_Layer.append(Convolution1D(filters=filters,
                                               kernel_size=kernel_size,
                                               padding='valid',
                                               activation=self.c['cnnactivate'],
                                               kernel_initializer=self.c['cnninitial']))
            MaxPool_Layer.append(MaxPooling1D(pool_size=int(self.c['sentencepad']-kernel_size+1)))
            Flatten_Layer.append(Flatten())
        Convolted_tensor0=[]
        Convolted_tensor1=[]
        for channel in range(len(self.c['cnnfilters'])):
            Convolted_tensor0.append(Convolt_Layer[channel](input0))
            Convolted_tensor1.append(Convolt_Layer[channel](input1))
        MaxPooled_tensor0=[]
        MaxPooled_tensor1=[]
        for channel in range(len(self.c['cnnfilters'])):
            MaxPooled_tensor0.append(MaxPool_Layer[channel](Convolted_tensor0[channel]))
            MaxPooled_tensor1.append(MaxPool_Layer[channel](Convolted_tensor1[channel]))
        Flattened_tensor0=[]
        Flattened_tensor1=[]
        for channel in range(len(self.c['cnnfilters'])):
            Flattened_tensor0.append(Flatten_Layer[channel](MaxPooled_tensor0[channel]))
            Flattened_tensor1.append(Flatten_Layer[channel](MaxPooled_tensor1[channel]))
        if len(self.c['cnnfilters']) > 1:
            Flattened_tensor0=concatenate(Flattened_tensor0)
            Flattened_tensor1=concatenate(Flattened_tensor1)
        else:
            Flattened_tensor0=Flattened_tensor0[0]
            Flattened_tensor1=Flattened_tensor1[0]
        absDifference = Lambda(lambda X:K.abs(X[0] - X[1]))([Flattened_tensor0,Flattened_tensor1])
        mulDifference = multiply([Flattened_tensor0,Flattened_tensor1])
        allDifference = concatenate([absDifference,mulDifference])
        for ilayer, densedimension in enumerate(self.c['densedimension']):
            allDifference = Dense(units=int(densedimension), 
                                  activation=self.c['denseactivate'], 
                                  kernel_initializer=self.c['denseinitial'])(allDifference)
        output = Dense(name='output',
                       units=self.c['num_classes'],
                       activation='softmax', 
                       kernel_initializer=self.c['denseinitial'])(allDifference)
        self.model = Model(inputs=[input0,input1], outputs=output)
        self.model.compile(loss={'output': self._lossfunction}, optimizer=self.c['optimizer'])
    def _lossfunction(self,y_true,y_pred):
        ny_true = y_true[:,1] + 2*y_true[:,2] + 3*y_true[:,3] + 4*y_true[:,4] + 5*y_true[:,5]
        ny_pred = y_pred[:,1] + 2*y_pred[:,2] + 3*y_pred[:,3] + 4*y_pred[:,4] + 5*y_pred[:,5]
        my_true = K.mean(ny_true)
        my_pred = K.mean(ny_pred)
        var_true = (ny_true - my_true)**2
        var_pred = (ny_pred - my_pred)**2
        return -K.sum((ny_true-my_true)*(ny_pred-my_pred),axis=-1) / (K.sqrt(K.sum(var_true,axis=-1)*K.sum(var_pred,axis=-1)))

    def eval_model(self):
        global TEXT_EVAL

        results = []
        for data in [self.traindata, self.validdata, self.testdata, self.track3, self.track4a, self.track4b, self.track5]:
            predictionclasses = []
            for dataslice,_ in self._sample_pairs(data, len(data['classes']), shuffle=False, once=True):
                predictionclasses += list(self.model.predict(dataslice))
            prediction = np.dot(np.array(predictionclasses),np.arange(self.c['num_classes']))
            goldlabels = data['labels']
            result=pearsonr(prediction, goldlabels)[0]
            results.append(round(result,4))
        print(TEXT_EVAL,end='')
        print(results)

        return tuple(results)


    def fit_model(self, wfname):
        kwargs = dict()
        kwargs['generator']       = self._sample_pairs(self.traindata, self.c['batch_size'])
        kwargs['steps_per_epoch'] = self.c['num_batchs']
        kwargs['epochs']          = self.c['num_epochs']
        class Evaluate(Callback):
            def __init__(self, task, wfname):
                self.task       = task
                self.bestresult = 0.0
                self.wfname     = wfname
            def on_epoch_end(self, epoch, logs={}):
                _,validresult,_,_,_,_,_  = self.task.eval_model()
                if validresult > self.bestresult:
                    self.bestresult = validresult
                    self.task.model.save(self.wfname)
        kwargs['callbacks'] = [Evaluate(self, wfname)]
        return self.model.fit_generator(verbose=2,**kwargs)
    def _sample_pairs(self, data, batch_size, shuffle=True, once=False):
        num = len(data['classes'])
        idN = int((num+batch_size-1) / batch_size)
        ids = list(range(num))
        while True:
            if shuffle: random.shuffle(ids)
            datacopy= copy.deepcopy(data)
            for name, value in datacopy.items():
                valuer=copy.copy(value)
                for i in range(num):
                    valuer[i]=value[ids[i]]
                datacopy[name] = valuer
            for i in range(idN):
                sl  = slice(i*batch_size, (i+1)*batch_size)
                dataslice= dict()
                for name, value in datacopy.items():
                    dataslice[name] = value[sl]
                x = [dataslice['m0'],dataslice['m1']]
                y = dataslice['classes']
                yield (x,y)
            if once: break

c = dict()
c['num_runs']   = 3
c['num_epochs'] = 64
c['num_batchs'] = 2
c['batch_size'] = 1024
c['wordvectdim']  = 300
c['sentencepad']  = 60
c['num_classes']  = 6
c['cnnfilters']     = {1: 1800}
c['cnninitial']     = 'he_uniform'
c['cnnactivate']    = 'relu'
c['densedimension'] = list([1800])
c['denseinitial']   = 'he_uniform'
c['denseactivate']  = 'tanh'
c['optimizer']  = 'adam'


def main():
    global ADD_PREFIX
    global TEXT_EVAL
    tsk = STSTask(c)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-lg','--train_language', type=str, default='en')
    parser.add_argument('-p', '--add_lang_prefix', action='store_true')
    args = parser.parse_args()

    ADD_PREFIX = args.add_lang_prefix

    tsk.load_resc(args.input)

    if args.train_language == 'en':
        tsk.load_data('STSdataset/sts-train.csv', 'STSdataset/sts-dev.csv', 'STSdataset/sts-test.csv',
                      'STSdataset/track3es-es.csv', 'STSdataset/track4aes-en.csv', 'STSdataset/track4bes-en.csv',
                      'STSdataset/track5en-en.csv')
    elif args.train_language == 'es':
        tsk.load_data('STSdataset/sts-train-es-split.csv', 'STSdataset/sts-dev-es-split.csv', 'STSdataset/track3es-es.csv',
                      'STSdataset/sts-test.csv', 'STSdataset/track4aes-en.csv', 'STSdataset/track4bes-en.csv',
                      'STSdataset/track5en-en.csv')
        TEXT_EVAL = '[Train, Valid, Test, test-en, track4aes-en, track4bes-en, track5en-en]='

    elif args.train_language == 'enes':
        tsk.load_data('STSdataset/sts-train-enes.csv', 'STSdataset/sts-dev-enes.csv', 'STSdataset/sts-test-enes.csv',
                      'STSdataset/sts-test.csv', 'STSdataset/track3es-es.csv', 'STSdataset/track4aes-en.csv',
                      'STSdataset/track4bes-en.csv')
        TEXT_EVAL = '[Train, Valid, Test, test-en, test-es, track4aes-en.csv, track4bes-en.csv]='

    else:
        raise ValueError('Language ' + args.train_language + ' not supported yet')

    bestresult = 0.0
    bestwfname = None
    for i_run in range(tsk.c['num_runs']):
        print('RunID: %s' %i_run)
        tsk.create_model()
        print('Training')
        wfname = './weightfile'+str(i_run)
        tsk.fit_model(wfname)
        print('Prediction(best valid epoch)')
        tsk.model.load_weights(wfname)
        _,validresult,_,_,_,_,_ = tsk.eval_model()
        if validresult>bestresult:
            bestresult = validresult
            bestwfname = wfname
    print('Prediction(best run)')
    tsk.model.load_weights(bestwfname)

    traindata, validdata, testdata, track3, track4a, track4b, track5 = tsk.eval_model()

    if not os.path.exists('Results'):
        os.makedirs('Results')

    output_name = 'Results/Results_STS_CNN_en.csv' if args.train_language == 'en' else ('Results/Results_STS_CNN_es.csv' if args.train_language=='es' else 'Results/Results_STS_CNN_enes.csv')

    with open(output_name, 'a+') as file:

        print(args.input + ',' + str(traindata) + ',' + str(validdata) + ',' + str(testdata) + ','
              + str(track3) + ',' + str(track4a) + ',' + str(track4b) + ',' + str(track5), file=file)

    print('Results exported to ' + str(output_name))


if __name__ == "__main__":
    main()