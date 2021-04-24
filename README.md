# Contextuality of Code Representation Learning Models: A Case Study of Bug Detection

<p aligh="center"> This repository is for <b>Contextuality of Code Representation Learning Models: A Case Study of Bug Detection</b> </p>

## Introduction ##

Word embedding models in natural language processing (NLP) have been
recently used as code representation learning (CRL) for machine
learning-based software engineering tasks. While some earlier models
produce static vectors (i.e., each word has a single vector regardless
of context), more advanced models produce contextualized vector
representations (i.e., the context-sensitive embeddings). Due to the
different nature of code and texts, we have conducted an extensive
empirical study to evaluate whether the word embedding models produce
static or contextualized vector representations for source code. As a
case study, we also evaluate whether the contextualized vectors help a
machine-learning-based bug detection model improve results.  We
conducted our experiments on a total of 12 sequence-
/tree-/graph-based embedding models and a dataset of 10,222 Java
projects with +14M methods. We report several interesting findings
including that the vectors from some NLP models do not reflect well
the contextuality nature of program units and that the contextualized
vectors help a model perform better than the static vectors in bug
detection. We also call for actions in developing code-oriented CRL models. 

----------

## Contents

1. [Dataset](#dataset)
2. [Approaches](#approaches)
3. [Requirements](#requirements)
4. [Process](#process)

## Dataset

The Dataset we used in the paper:

For the Contextuality of Code Representation Evaluation:

Code2vec: ```ttps://github.com/tech-srl/code2vec```
  
For Bug Detection Case study:

BigFix: ```https://drive.google.com/open?id=1KL3M-BbisVLWXyvn05V6huSLNUby_9qN```

## Approaches

Here are the sequence-based approaches:

Word2vec: ```https://github.com/tmikolov/word2vec```

GloVe: ```https://github.com/stanfordnlp/GloVe```

FastText: ```https://github.com/facebookresearch/fastText```

ELMo: ```https://allennlp.org/elmo```

Bert: ```https://github.com/google-research/bert```

Here are the tree-based approaches:

Tree-LSTM: ```https://github.com/stanfordnlp/treelstm```

ASTNN: ```https://github.com/zhangj111/astnn```

TBCNN: ```https://sites.google.com/site/treebasedcnn/```

TreeCaps: ```https://github.com/vinojjayasundara/treecaps```

Here are the graph-based approaches:

Node2vec: ```https://github.com/aditya-grover/node2vec```

Deepwalk: ```https://github.com/phanein/deepwalk```

Graph2vec: ```https://github.com/MLDroid/graph2vec_tf```

## Requirements

See ```Requirement.txt``` for the required packages and environment for each baseline.

## Process

1. Clone the repository ```git clone https://github.com/VectorEvaluation/Vector-Evaluation-Contextuality.git``` 

2. Download both two datasets and put in the folder ```.\data```.

3. To run the evaluation on RQ1-RQ4, please run ```python3 RQx```, ```x``` is the RQ number that you want to run.

Note: We use the virtural environment to switch the required environments for each baseline. You may can use other ways to do so, but you need to change the environment switch code for each RQ.

