# MVM-Embeddings (Meta-embeddings VecMap)

This repository contains an open source implementation of our framework to learn and evaluate mono-lingual and cross-lingual meta-embeddings. 

## Requeriments

* numpy
* sklearn
* scipy
* tensorflow (2.0.0)

## Usage

Our code assumes the word2vec.txt format for all the embeddings, that is, the embeddings should stored as a text file with a header including the number of words in the word embedding followed by the number of dimensions of the vectors
```
10000 300
person 3.75 1.23 2.69 3.45
make 1.17 4.62 -1.43 2.57
```
If your embeddings are in the glove.txt format (without the header) you can use utils/embedding_converted.py (i.e python3 embedding_converted.py -i myEmb.emb -w2v outEmb.vec). This script is also useful if your embeddings include duplicated words or other formating problems that cause another script to crash. 


### Monolingual meta-embeddings
If you want to generate a meta-embedding combining monolingual source embeddings (all in the same language) you can do it with just 1 command:
```
python3 -i source_embedding_1.vec source_embedding_2.vec source_embedding_3.vec -t source_embedding_1.vec -o meta_embedding.vec
``` 
Parameters:
 * -i: List of source embeddings, you can use as much as you want!!
 * -t: The vector space to which all the source embeddings will be mapped using VecMap. It can be any of the source embeddings or a different word embedding. Generally any word embedding with a large vocabulary will be fine. 
 * -o: Path where the meta embedding will be written.
 * -k: Number of nearest neighbors to use in the OOV generation step (default: 10). 
 * -v: If you just want to generate a meta embedding containing a list of word stored in a text file (a word per line) use this parameter followed by the path of the file.
 * -b: batch size. Increasing this value helps the OOV step to run faster. If you get Out of Memory errors lower it. (Default: 256, should be fine for a 8GB GPU)
 * -r: If you want to improve the source embeddings using retrofitting you can and this parameter followed by the lexicon that you want to use (more info: https://github.com/mfaruqui/retrofitting). Not recommended in general, but it may improve the result in some task.
 * -rn: Retrofitting number of iterations (default:10)
 * -nc: Do not delete the intermediate files (outputs from Vecmap). Use it if you love to store a lot of useless big files in your hard drive. 
## References
This repository includes code from other authors, check out their work if you use our software!

* Vecmap: Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), pages 5012-5019. https://github.com/artetxem/vecmap

* Retrofitting: Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015). Retrofitting Word Vectors to Semantic Lexicons. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1606-1615). https://github.com/mfaruqui/retrofitting

* STS-CNN: Yang Shao. 2017. HCTI at SemEval-2017 Task 1: Use convolutional neural network to evaluate semantic textual similarity. In Proceedings of SemEval-2017. https://github.com/celarex/Semantic-Textual-Similarity-STSbenchmark-HCTI-CNN

* Some code has also been adapted from "Word Embeddings Benchmarks": https://github.com/kudkudak/word-embeddings-benchmarks
