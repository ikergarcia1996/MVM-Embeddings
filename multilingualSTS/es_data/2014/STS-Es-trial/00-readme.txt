
			SemEval 2014 Task

	       Semantic Textual Similarity for Spanish:

			     TRIAL DATASET
				   


This set of files pertains to the TRIAL DATASET for the Semantic Textual
Similarity for Spanish task in SemEval-2014.

The trial dataset contains the following:

  00-README.txt 		  this file

  STS.input.li65.txt         tab separated sample input file with 
                                  sentence pairs

  STS.gs.li65.txt            gold standard for the trial data
  
  STS.output.li65.txt        expected system output format
  
  correct-output.pl          script that ensures the correctness of system
                                  output format
 
  correlation.p              script that calculates the Pearson correlation
                                  between a system's predictions and the 
                                  corresponding gold standard


Introduction
------------

Similar to the semantic text similarity (STS) tasks conducted as part 
of SemEval 2012 and *Sem 2013, the participating systems are to 
predict how similar two sentences of text, s1 and s2, are, by returning a
similarity score.

The trial dataset comprises a set of 65 pairs of sentences in Spanish, 
which can be used to develop and train systems.

This trial data is similar to the sentence pairs that will be
used as test data. The associated gold standard scores represent the average
similarity score provided by three native speakers of Spanish. The goal of 
these samples is to allow participants to have a clearer idea on what the
task entails, and what type of data can be expected. 

The sentence pairs have been manually tagged with a number from 0 to
5, as defined below (cf. Gold Standard section). 

Input format
------------

The input file consist of two fields separated by tabs (see STS.input.li65.txt):

- first sentence (does not contain tabs)
- second sentence (does not contain tabs)


Gold Standard
-------------

The gold standard contains a score between 0 and 4 for each pair of
sentences, with the following interpretation:

(4) The two sentences are completely equivalent, as they mean the same
    thing.  

      The bird is bathing in the sink.  
      Birdie is washing itself in the water basin.

(3) The two sentences are roughly equivalent, but some information differs/ 
    is missing.
      
      John said he is considered a witness but not a suspect.
      "He is not a suspect anymore." John said.

(2) The two sentences are not equivalent, but share some details.

      They flew out of the nest in groups.
      They flew into the nest together. 

(1) The two sentences are not equivalent, but are on the same topic.

      The woman is playing the violin.
      The young lady enjoys listening to the guitar.

(0) The two sentences are completely unrelated.
      John went horse back riding at dawn with a whole group of friends.
      Sunrise at dawn is a magnificent view to take in if you wake up
      early enough for it.

Format: the gold standard file consist of one single field per line (see
STS.gs.li65.txt):

- a number between 0 and 4


Answer format
--------------

The answer format is follows the gold standard format:
- a number between 0 and 4 (the similarity score)

The output file needs to conform to the above specifications. Files
which do not follow those will be automatically removed from
evaluation. Please check that your answer files are in the correct
format using the following script:
 
  $ ./correct-output.pl STS.output.li65.txt
  Output file is OK!

In addition to printing errors and a final message on standard error,
the script returns 0 if correct, and another value if incorrect. 



Scoring
-------

The official score is based on the average of Pearson correlation. For instance the following script returns the correlation for individual pairs:

  $ ./correlation.pl STS.gs.li65.txt  STS.output.li65.txt
  Pearson: 1


Participation in the task
-------------------------

Participant teams will be allowed to submit three runs at most.


Other
-----

Please check http://alt.qcri.org/semeval2014/task10/ for more details.



Authors
-------

Carmen Banea
Claire Cardie
Rada Mihalcea
Janyce Wiebe



Notes
-----
This readme file is based on the file with the same name distributed as part of the STS English tasks (SemEval 2012 and *Sem 2013). The general task format is maintained in order to encourage a unified framework across the evaluations. The scripts developed as part of those tasks are also reused here.


