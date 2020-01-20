# MVM-Embeddings (Meta-embeddings VecMap)

This repository contains an open-source implementation of our framework to learn and evaluate mono-lingual and cross-lingual meta-embeddings. 

## Requeriments

* Python 3
* numpy
* sklearn
* scipy
* tensorflow (2.0.0)

## Citation
´´´
@misc{garca2020common,
    title={A Common Semantic Space for Monolingual and Cross-Lingual Meta-Embeddings},
    author={Iker García and Rodrigo Agerri and German Rigau},
    year={2020},
    eprint={2001.06381},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

## Usage
´´´
Our code assumes the word2vec.txt format for all the embeddings, that is, the embeddings should be stored as a text file with a header including the number of words in the word embedding followed by the number of dimensions of the vectors
```
10000 300
person 3.75 1.23 2.69 3.45
make 1.17 4.62 -1.43 2.57
```
If your embeddings are in the glove.txt format (without the header) you can use utils/embedding_converted.py (i.e python3 embedding_converted.py -i myEmb.emb -w2v outEmb.vec). This script is also useful if your embeddings include duplicated words or other formating problems that cause another script to crash. 


### Monolingual meta-embeddings
If you want to generate a meta-embedding combining monolingual source embeddings (all in the same language) you can do it with just 1 command:
```
python3 generate_mvm_embeddings.py -i source_embedding_1.vec source_embedding_2.vec source_embedding_3.vec -t source_embedding_1.vec -o meta_embedding.vec
``` 
Parameters:
 * -i: List of source embeddings, you can use as many as you want!!
 * -t: The vector space to which all the source embeddings will be mapped using VecMap. It can be any of the source embeddings or a different word embedding. Generally, any word embedding with a large vocabulary will be fine. 
 * -o: Path where the meta embedding will be written.
 * -k: Number of nearest neighbours to use in the OOV generation step (default: 10). 
 * -v: If you just want to generate a meta embedding containing a list of words stored in a text file (a word per line) use this parameter followed by the path of the file.
 * -b: batch size. Increasing this value helps the OOV step to run faster. If you get Out of Memory errors lower it. (Default: 256, should be fine for an 8GB GPU)
 * -r: If you want to improve the source embeddings using retrofitting you can and this parameter followed by the lexicon that you want to use (more info: https://github.com/mfaruqui/retrofitting). Not recommended in general, but it may improve the result in some tasks.
 * -rn: Retrofitting number of iterations (default:10)
 * -oov: Do not use the OOV generation algorithm (NN). 
 * -nc: Do not delete the intermediate files (outputs from Vecmap). Use it if you love to store a lot of useless big files in your hard drive. 


Real usage example:
```
python3 generate_mvm_embeddings.py -i Embeddings/crawl-300d-2M.vec Embeddings/UKBV_L2_header.vec Embeddings/en_AT.txt Embeddings/Param.vec -t Embeddings/crawl-300d-2M.vec -o MetaEmbeddings/FT_UKBV_AT_Param.FT.vec
``` 

Alternatively, you can use average, concatenation and concatenation + dimensionality reduction instead of our method to generate meta-embedings. These baseline methods are worse but they can be useful for performance comparisons. Average and Concatenation will also be done using our OOV generation approach (unless the -oov flag is specified). All the vectors will be length so all the source embeddings contribute the same to the final meta embedding.  
```
python3 embeddings_concatenate.py -i source_embedding_1.vec source_embedding_2.vec source_embedding_3.vec -o concat_meta_embedding.vec
python3 dimensionality_reduction.py -i concat_meta_embedding.vec -m [PCA, tSVD, DRA <- default] -n 300 -o concat_dimre_meta_embedding.vec
python3 embeddings_mean.py -i source_embedding_1.vec source_embedding_2.vec source_embedding_3.vec -o avg_meta_embedding.vec
``` 

### Multilingual meta-embeddings

For multilingual meta-embeddings one extra step should be done. Fist of all remember that we need at least one cross-lingual word embedding, that is, if you want to generate a Spanish-English cross-lingual meta-embedding you need at least one already Spanish-English cross-lingual source embedding. Apart from this constraint, you can use as many monolingual and cross-lingual source embeddings as you want.

Many cross-lingual word embeddings include a prefix that indicates the language of the word. For example "en/person". It may be a good idea to add to the monolingual embeddings these prefixes, removing it from the cross-lingual embeddings can cause problems with duplicated words since many languages have many words in common. Many scrips may be useful to manage this in the "utils" folder:
```
Add a prefix to all the words in a word embedding (i.e /en prefix) 
python3 add_prefix.py english_embedding.vec -o english_embedding.prefix.vec -p en

Remove all prefixes from embedding (asumes / separator)
python3 delete_prefix.py english_embedding.prefix.vec -o english_embedding.vec

Given a cross-lingual word embedding generates a new embedding containing just the words with a given prefix (extract words in a language)
python3 extract_language.py crosslingual_embeddings.vec -o english_embedding.vec -p en

Merge two embeddings (print one after the other)
python3 merge.py -i embedding_1.vec embedding_2.vec embedding_3.vec -o merged.vec
```

After you have all your source word embedding all in the same format we will first perform the OOV generation step. We should do this first because if we have two monolingual source embeddings in different languages since they don't have any vocabulary in common the program will crash. This step what it does is, using an already cross-lingual embedding, transforms every monolingual embedding in a cross-lingual embedding, this way all the source embeddings will have some vocabulary in common. We will do this step with all of our monolingual source embeddings. We will use the generate_words.py file in the "utils" folder. 
```
python3 generate_words.py -i monolingual_embedding_lang1.vec -c crosslingual_embedding_lang1lang2 -o monolingual_embedding.extended.vec
```
Parameters
* -i: path of the mono-lingual embedding.
* -c: path of the cross-lingual embedding. 
* -b: batch size. Increasing this value helps the OOV step to run faster. If you get Out of Memory errors lower it. (Default: 1024, should be fine for an 8GB GPU)
* -k: Number of nearest neighbours to use in the OOV generation step (default: 10). 

Real example
```
python3 generate_words.py -i Embeddings/crawl-300d-2M.vec -c Embeddings/Joint.txt -o Embeddings/extended/crawl-300d-2M.vec
```

After this step, you can generate your meta embeddings in the same way that we do for the monolingual meta-embeddings. Just take into account that you probably want to use one of the cross-lingual embeddings as the vector space to map all the source embeddings (-t). 
A real example to generate an English-Spanish cross-lingual meta-embedding. 
``` 

python3 generate_words.py -i Embeddings/en_AT.txt -c Embeddings/Joint.txt -o Embeddings/extended/en_AT.txt
python3 generate_words.py -i Embeddings/fasttext-sbwc.3.6.e20.vec -c Embeddings/Joint.txt -o Embeddings/extended/fasttext-sbwc.3.6.e20.vec
python3 generate_words.py -i Embeddings/wiki-news-300d-1M.vec -c Embeddings/Joint.txt -o Embeddings/extended/wiki-news-300d-1M.vec
python3 generate_words.py -i Embeddings/crawl-300d-2M.vec -c Embeddings/Joint.txt -o Embeddings/extended/crawl-300d-2M.vec
python3 generate_words.py -i Embeddings/Param.vec -c Embeddings/Joint.txt -o /Embeddings/extended/Param.vec


python3 generate_mvm_embeddings.py -i Embeddings/Joint.txt Embeddings/numberbatch.txt /Embeddings/extended/fasttext-sbwc.3.6.e20.vec Embeddings/extended/crawl-300d-2M.vec Embeddings/extended/en_AT.txt Embeddings/extended/Param.vec -t /Embeddings/Joint.txt -o enes_extended_Joint_numberbatch_FTes_FTen_AT_Param.Joint.vec
``` 

## Evaluation
You can evaluate any embedding and meta-embedding in the word similarity task and in STSbenchmarks (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) task using the CNN model from Yang Shao (https://github.com/celarex/Semantic-Textual-Similarity-STSbenchmark-HCTI-CNN)

``` 
python3 evaluate_similarity -i embedding.vec -lg en 
``` 
Parameters:
 * -i: path to the embedding that you want  to evaluate
 * -d: A directory containing multiple word embeddings that you want to evaluate (mutually exclusive with -i)
 * -l: lowercase all the words in the datasets
 * -lg: language in which you want to evaluate word similarity. Options 'en': English. 'es': Spanish. 'enes': English-Spanish cross-lingual task. You can include multiple languages, for example "-lg en es enes".
 * -p: Use it if the words have a prefix indicating the language (i.e 'en/person'). Assumes 'en' for English and 'es' por Spanish.

``` 
python3 evaluate_STS_CNN,py -i embedding.vec -lg en 
``` 
 Parameters
  * -i: path to the embedding that you want  to evaluate
  * -lg: language in which you want to train the model. Options 'en': English. 'es': Spanish. 'enes': English-Spanish dataset. 
  * -p: Use it if the words have a prefix indicating the language (i.e 'en/person'). Assumes 'en' for English and 'es' for Spanish.
 
## References
This repository includes code from other authors, check out their work if you use our software!

* Vecmap: Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), pages 5012-5019. https://github.com/artetxem/vecmap

* Retrofitting: Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015). Retrofitting Word Vectors to Semantic Lexicons. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1606-1615). https://github.com/mfaruqui/retrofitting

* STS-CNN: Yang Shao. 2017. HCTI at SemEval-2017 Task 1: Use convolutional neural network to evaluate semantic textual similarity. In Proceedings of SemEval-2017. https://github.com/celarex/Semantic-Textual-Similarity-STSbenchmark-HCTI-CNN

* DRA: Mu, J., Bhat, S., & Viswanath, P. (2017). All-but-the-top: Simple and effective postprocessing for word representations. arXiv preprint arXiv:1702.01417. https://github.com/vyraun/Half-Size

* Some code has also been adapted from "Word Embeddings Benchmarks": https://github.com/kudkudak/word-embeddings-benchmarks

## License
See [License](https://github.com/ikergarcia1996/MVM-Embeddings/blob/master/LICENSE)
Note: This license only applies for our code. External APIs (see [References](#references)) may have a different license.
