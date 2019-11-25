# This file used code from: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
import torch
import sys
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel
from similarity_datasets import get_vocab_all
from utils import printTrace
import datetime


printTrace("Loading vocabulary from datasets...")
vocab = get_vocab_all(lower=True)
printTrace("Done!")
printTrace("Loading BERT...")
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
printTrace("Done!")
concat_bert = open("../Embeddings/BERT.concat", 'w+')
sum_bert = open("../Embeddings/BERT.sum", 'w+')
avg_bert = open("../Embeddings/BERT.avg", 'w+')
last_bert = open("../Embeddings/BERT.last", 'w+')

for i, word in enumerate(vocab):
    if i % 100 == 0:
        string = "<" + str(datetime.datetime.now()) + ">  " + 'Generating static word representations from BERT: ' + \
                 str(int(100 * i / len(vocab))) + '%'
        print(string, end="\r")

    text = "[CLS] " + str(word) + " [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    token_embeddings = []
    for token_i in range(len(tokenized_text)):
        hidden_layers = []
        for layer_i in range(len(encoded_layers)):
            vec = encoded_layers[layer_i][0][token_i]
            hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

    sentence_embedding = torch.mean(encoded_layers[11], 1)
    print(str(word) + ' ' + ' '.join(['%.6g' % x for x in sentence_embedding[0]]), file=avg_bert)

    concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings]
    sentence_embedding = torch.mean(torch.stack(concatenated_last_4_layers), 0)
    print(str(word) + ' ' + ' '.join(['%.6g' % x for x in sentence_embedding]), file=concat_bert)

    sum_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
    sentence_embedding = torch.mean(torch.stack(sum_last_4_layers), 0)
    print(str(word) + ' ' + ' '.join(['%.6g' % x for x in sentence_embedding]), file=sum_bert)

    last_layer = [layer[-1] for layer in token_embeddings]
    sentence_embedding = torch.mean(torch.stack(last_layer), 0)
    print(str(word) + ' ' + ' '.join(['%.6g' % x for x in sentence_embedding]), file=last_bert)

print()
printTrace("Done!")
printTrace("Evaluating word similarity...!")
com = 'python3 evaluate_similarity.py -i ../Embeddings/BERT.sum -lg en -l'
print(com)
os.system(com)
com = 'python3 evaluate_similarity.py -i ../Embeddings/BERT.concat -lg en -l'
print(com)
os.system(com)
com = 'python3 evaluate_similarity.py -i ../Embeddings/BERT.avg -lg en -l'
print(com)
os.system(com)
com = 'python3 evaluate_similarity.py -i ../Embeddings/BERT.last -lg en -l'
print(com)
os.system(com)
printTrace("Done!")
