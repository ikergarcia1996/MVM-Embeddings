import os

data_dir = 'es_data/2014/STS-Es-trial/'

def main():
    for file in os.listdir(data_dir):
        if 'input' in file:
            gold = file.replace('input', 'gs')
            name = ''.join(file.split('.')[2:4]) + '.csv'
            data = open(data_dir + file, encoding='utf8').readlines()
            gs = open(data_dir + gold, encoding='utf8').readlines()

            with open(name, 'w+', encoding='utf8') as output:
                for i in range(len(data)):
                    print(name.split('.')[0] + '\tNA\t2017\t' + str(i) + '\t' + gs[i].rstrip() + '\t' + data[i].rstrip(), file=output)


if __name__ == "__main__":
    main()
