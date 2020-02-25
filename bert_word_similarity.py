import torch
from transformers import *
from similarity_datasets import get_vocab_all
import numpy as np
import os
import datetime
import argparse


def bert_ws(output_path: str, model_name: str = "bert-base-cased", num_last_layers = 13):

    files = [
        open(os.path.join(output_path, f"{i}.vec"), "w+", encoding="utf-8")
        for i in range(num_last_layers)
    ]
    files_avg = [
        open(os.path.join(output_path, f"avg_from_{i}.vec"), "w+", encoding="utf-8")
        for i in range(num_last_layers)
    ]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(
        model_name, output_hidden_states=True, output_attentions=True
    )
    model.eval()
    model.to(device="cuda:0")
    vocab = get_vocab_all(lower=True)

    for word_no, word in enumerate(vocab):

        if word_no % 1 == 0:
            string = (
                "<"
                + str(datetime.datetime.now())
                + ">  "
                + "Generating static word representations from BERT: "
                + str(int(100 * word_no / len(vocab)))
                + "%"
            )
            print(string, end="\r")

        input_ids = torch.tensor(
            [tokenizer.encode(f"[CLS] {word} ", add_special_tokens=False)]
        )

        with torch.no_grad():
            all_hidden_states, _ = model(input_ids.to(device="cuda:0"))[-2:]

        # print layers

        hidden_states = np.asarray(
            [
                all_hidden_states[layer][0][1:].cpu().numpy()
                for layer in range(num_last_layers)
            ]
        )
        hidden_states = np.asarray([np.average(x, axis=0) for x in hidden_states])

        for layer in range(num_last_layers):
            print(
                word + " " + " ".join(str(x) for x in hidden_states[layer]),
                file=files[layer],
            )

        # print avg layers
        for layer in range(num_last_layers):
            avg_vectors = np.asarray(hidden_states[:layer])
            avg = np.average(avg_vectors, axis=0)

            print(
                word + " " + " ".join(str(x) for x in avg), file=files_avg[layer],
            )

    [f.close() for f in files]
    [f.close() for f in files_avg]

    for i in range(num_last_layers):
        filename = os.path.join(output_path, f"{i}.vec")
        com = f"python3 evaluate_similarity.py -i {filename} -lg en -l"
        os.system(com)

    for i in range(num_last_layers):
        filename = os.path.join(output_path, f"avg_from_{i}.vec")
        com = f"python3 evaluate_similarity.py -i {filename} -lg en -l"
        os.system(com)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    bert_ws(output_path=args.output_dir, model_name=args.model_name)
