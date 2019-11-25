from sklearn.model_selection import train_test_split

dataset = '../STSdataset/sts-train-es.csv'

lines = open(dataset,'r',encoding='utf8').readlines()

train,dev = train_test_split(lines,train_size=0.8)


with open('../STSdataset/sts-train-es-split.csv','w', encoding='utf8') as file:
    for line in train:
        print(line.strip(), file=file)

with open('../STSdataset/sts-dev-es-split.csv', 'w', encoding='utf8') as file:
    for line in dev:
        print(line.strip(), file=file)

