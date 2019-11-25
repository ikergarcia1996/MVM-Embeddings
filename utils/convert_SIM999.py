path = '../SimilarityDataset/ES-SIM999.txt'


pairs = []
with open(path,'r') as file:
    for line in file:
        a, b, g = line.split(';')[1:-1]
        pairs.append(a+'\t'+b+'\t'+str(g))

with open(path,'w+') as file:
    for line in pairs:
        print(line,file=file)
