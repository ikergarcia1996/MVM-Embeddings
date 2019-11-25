from sklearn.datasets.base import Bunch
import pandas as pd
import numpy as np
from six import iteritems

dataset_path = 'SimilarityDataset/'

Mturk287_path = dataset_path + 'EN-MTurk-287.txt'
Mturk771_path = dataset_path + 'EN-MTurk-771.txt'
mendev_path = dataset_path + 'EN-MEN-LEM-DEV.txt'
mentest_path = dataset_path + 'EN-MEN-LEM-TEST.txt'
men_path = dataset_path + 'EN-MEN-LEM.txt'
WS353_all_path = dataset_path + 'EN-WS353_all.txt'
WS353_relaredness_path = dataset_path + 'EN-WSR353_relatedness.txt'
WS353_similarity_path = dataset_path + 'EN-WSS353_similarity.txt'
WS353_set1_path = dataset_path + 'EN-WS353-SET1.txt'
WS353_set2_path = dataset_path + 'EN-WS353-SET2.txt'
RG65_path =dataset_path + 'EN-RG-65.txt'
RW_path = dataset_path + 'EN-RW.txt'
SIM999_path = dataset_path + 'EN-SIM999.txt'
Verb143_path = dataset_path + 'EN-VERB-143.txt'
MC30_path = dataset_path + 'EN-MC-30.txt'
YP_130_path = dataset_path + 'EN-YP-130.txt'
SimVerb3500_path = dataset_path + 'EN-SimVerb-3500.txt'
WS353_ES_all_path = dataset_path + 'ES-WS353_all.txt'
RG65_ES_path = dataset_path +'ES-RG-65.txt'
RG65_ENES_path = dataset_path +'EN-ES-RG-65.txt'
WS353_ENES_all_path = dataset_path + 'EN-ES-WS353_all.txt'
SIM999_ENES_path = dataset_path + 'ENES-SIM999.txt'
SemEval_EN_path = dataset_path + 'EN-SemEval.txt'
SemEval_ES_path = dataset_path + 'ES-SemEval.txt'
SemEval_ENES_path = dataset_path + 'EN-ES-SemEval.txt'
SIM999_ES_path = dataset_path + 'ES-SIM999.txt'

def get_MTurk287():
    data = pd.read_csv(Mturk287_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=2 * data[:, 2].astype(np.float))


def get_MTurk771():
    data = pd.read_csv(Mturk771_path, sep=" ", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=2 * data[:, 2].astype(np.float))


def get_MENdev():
    data = pd.read_csv(mendev_path, sep=" ", header=None)
    data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2].astype(np.float) / 5.0)


def get_MENtest():
    data = pd.read_csv(mentest_path, sep=" ", header=None)
    data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2].astype(np.float) / 5.0)


def get_MENall():
    data = pd.read_csv(men_path, sep=" ", header=None)
    data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2].astype(np.float) / 5.0)


def get_WS353_all():
    data = pd.read_csv(WS353_all_path, sep="\t", header=0).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_WS353_relatedness():
    data = pd.read_csv(WS353_relaredness_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_WS353_similarity():
    data = pd.read_csv(WS353_similarity_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_WS353_set1():
    data = pd.read_csv(WS353_set1_path, sep="\t", header=0).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_WS353_set2():
    data = pd.read_csv(WS353_set2_path, sep="\t", header=0).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_RG65():
    data = pd.read_csv(RG65_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def get_RW():
    data = pd.read_csv(RW_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_SimLex999():
    data = pd.read_csv(SIM999_path, sep="\t", header=0).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 3].astype(np.float))


def get_Verb143():
    data = pd.read_csv(Verb143_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0)


def get_MC30():
    data = pd.read_csv(MC30_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def get_YP130():
    data = pd.read_csv(YP_130_path, sep=" ", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def get_SimVerb3500():
    data = pd.read_csv(SimVerb3500_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_WS353_ES_all():
    data = pd.read_csv(WS353_ES_all_path, sep="\t", header=0, skiprows=1).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_RG65_ES():
    data = pd.read_csv(RG65_ES_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def get_RG65_ENES():
    data = pd.read_csv(RG65_ENES_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def get_WS353_ENES_all():
    data = pd.read_csv(WS353_ENES_all_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))


def get_SimLex999_ENES():
    data = pd.read_csv(SIM999_ENES_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))

def get_SimLex999_ES():
    data = pd.read_csv(SIM999_ES_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float))

def get_SemEval(lang='en'):
    data = None
    if lang=='en':
        data = pd.read_csv(SemEval_EN_path, sep="\t", header=None).values
    elif lang=='es':
        data = pd.read_csv(SemEval_ES_path, sep="\t", header=None).values
    elif lang=='enes':
        data = pd.read_csv(SemEval_ENES_path, sep="\t", header=None).values
    return Bunch(X=data[:, 0:2].astype("object"), y=data[:, 2].astype(np.float) * 10.0 / 4.0)


datasets_en = {
    "SimLex999": get_SimLex999(),
    "MTurk-287": get_MTurk287(),
    "MTurk-771": get_MTurk771(),
    "MEN_DEV": get_MENdev(),
    "MEN_TEST": get_MENtest(),
    "MEN_ALL": get_MENall(),
    "WS353_all": get_WS353_all(),
    "WS353_relatedness": get_WS353_relatedness(),
    "WS353_similarity": get_WS353_similarity(),
    "WS353_set1": get_WS353_set1(),
    "WS353_set2": get_WS353_set2(),
    "RG65": get_RG65(),
    "VERB-143": get_Verb143(),
    "MC-30": get_MC30(),
    "YP-130": get_YP130(),
    "SimVerb-3500": get_SimVerb3500(),
    "RW": get_RW(),
    "SemEval": get_SemEval(lang='en')

}

datasets_es = {
    "SL999": get_SimLex999_ES(),
    "WS353_all": get_WS353_ES_all(),
    "RG65": get_RG65_ES(),
    "SemEval": get_SemEval(lang='es')
}

datasets_enes = {
    "SL999": get_SimLex999_ENES(),
    "WS353_all": get_WS353_ENES_all(),
    "RG65": get_RG65_ENES(),
    "SemEval": get_SemEval(lang='enes')
}


def get_datasets(lang='en'):
    if lang == 'en':
        return datasets_en
    if lang == 'es':
        return datasets_es
    if lang == 'enes':
        return datasets_enes

    raise ValueError(str(lang) + ' not suported')


def get_vocab_all(lang='en', lower=False):
    vd = []
    for name, data in iteritems(get_datasets(lang)):
        vd = np.append(vd, np.append((data.X[:, 0]), (data.X[:, 1])))
    if lower:
        for i in range(len(vd)):
            vd[i] = vd[i].lower()
    return set(vd)


def get_dataset(dataset):
    if dataset == 'MTurk-287':
        return get_MTurk287()
    elif dataset == 'MTurk-771':
        return get_MTurk771()
    elif dataset == 'MEN':
        return  get_MENall()
    elif dataset == 'WS353_all':
        return  get_WS353_all()
    elif dataset == 'WS353_relatedness':
        return  get_WS353_relatedness()
    elif dataset == 'WS353_similarity':
        return  get_WS353_similarity()
    elif dataset == 'WS353_set1':
        return  get_WS353_set1()
    elif dataset == 'WS353_set2':
        return  get_WS353_set2()
    elif dataset == 'RG65':
        return  get_RG65()
    elif dataset == 'RW':
        return  get_RW()
    elif dataset == 'VERB-143':
        return  get_Verb143()
    elif dataset == 'MC-30':
        return  get_MC30()
    elif dataset == 'YP-130':
        return  get_YP130()
    elif dataset == 'SimVerb-3500':
        return  get_SimVerb3500()
    elif dataset == 'SimLex999':
        return  get_SimLex999()
    else:
        raise ValueError("The dataset {} is not supported".format(dataset))
