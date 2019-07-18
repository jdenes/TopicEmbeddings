# Topic Embeddings Framework
A code to create and test document embeddings using topic models. Based on forthcoming paper (maybe): "Document Embedding Using Topic Models", a Master Thesis at Sorbonne Université.

Topic Embeddings Framework is a Python 3 open-source code to generate and test document embeddings using most common topic models. It is designed to allow easier and centralized generation of such embeddings, and to test their performances against mainstream embeddings such as bag of word or doc2vec.

**Author :** [Julien Denes](https://github.com/jdenes/), Sciences Po médialab.

---

## Algorithms

### Included

Some embeddings using topic models are directly 

**References:**
* Bag of Words (abbreviated BOW) using gensim
* doc2vec (DOC2VEC) using gensim
* Word Embedding Agregation (POOL) using gensim's word2vec and mean as pooling function
* Bag of Random  Embedding  Projection (BOREP)
* Latent Semantic Analysis (LSA) using gensim

**Topic embeddings:**
* Latent Dirichlet Allocation (LDA) using gensim
* Hierarchical Dirichlet Process (HDP)
* Dynamic Topic Model (DTM) using gensim's wrapper
* Structural Topic Model (STM) using R package stm
* Correlated Topic Model (CTM) using R package topicmodels

### Supported

Unfortunatly, not all topic models are implented in Python or R. We support Pseudo-Document-Based Topic Model (PTM), which can be computed using using [STTM](https://github.com/qiang2100/STTM). Once you've done that, simply import the topic proportion file into `/external/` and name it `raw_embeddings/tmp_PTM_EMBEDDING_K.csv` where $K$ if your vector size.

## Requirements

* [Python 3.6+](https://www.python.org/downloads/) with packages in `requirements.txt`. To install them, simply run `pip install -r requirements.txt`.
* If you intend to use STM or CTM, you will also need to have [R](https://www.r-project.org/) installed. Required packages will install automatically.
* Please make sure that both `python` and `Rscript` can be run in your terminal no matter the directory you're in. If you're using Windows, consider adding them to your `PATH` environment variable ([see an example](https://datatofish.com/add-python-to-windows-path/)).

## Input data

To possible data inputs are available for now:
* You can use the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/), a popular standard data set for experiments in NLP,
* Or you can use your own data. If you wish to do so, you need to provide a `.csv` file with two mandatory columns: one called `text` containing your text, the other called `label` containing the document's label in numerical format. A column called `year` can be optionally included to be used in DTM or STM, in which case your data **must** be sorted according to this column. All other columns will be ignored, except by STM. An example file is provided: `datasets/example.csv`.

**File format of input corpus:**  Similar to file `corpus.txt` in the `data` folder, we assume that each line in the input corpus represents a document. Here, a document is a sequence of words/tokens separated by white space characters. The users should preprocess the input corpus before training the short text topic models, for example: down-casing, removing non-alphabetic characters and stop-words, removing words shorter than 3 characters and words appearing less than a certain times. Otherwise you may also implement a .csv ?

## Usage

Our algorithm runs in two steps. First, generate the embedding from the sata and save it. Second, run classification tests to assess the performance of your embeddings.

### Step 1: create embeddings from the data

**Train the algorithms to producein by executing:**

	$ python encode.py –model <LDA or BTM or PTM or SATM or DMM or WATM> -corpus <path/to/input_corpus_file.txt> [-ntopics <int>] [-niters <int>] [-twords <int>] [-name <String>]

where parameters in [ ] are optional. More parameters in different methods are shown in "src/utility/CmdArgs"

`-model`: Specify the topic model LDA or DMM

`-corpus`: Specify the path to the input corpus file.

`-ntopics <int>`: Specify the number of topics. The default value is 200.

`-nwords <int>`: Specify the number of the most frequent words to use (if applicable). The default value is 20.

`-name <String>`: Specify a name to the topic modeling experiment. The default value is `model`.

**Examples:**

	$ java -jar jar/STTM.jar -model BTM -corpus dataset/corpus.txt -name corpusBTM

The output files are saved in the "results" folder containing `corpusBTM.theta`, `corpusBTM.phi`, `corpusBTM.topWords`, `corpusBTM.topicAssignments` and `corpusBTM.paras` referring to the document-to-topic distributions, topic-to-word distributions, top topical words, topic assignments and model parameters, respectively. 

### Step 2: use your embedding(s) on a classification task

**Additional input:** your file with labels.

**Usage:**

	$ python classify.py –model True –label <label_file_path> -prob <Document-topic-prob/Suffix>

`–label`: Specify the path to the ground truth label file. Each line in this label file contains the golden label of the corresponding document in the input corpus. See files `corpus.LABEL` and `corpus.txt` in the `dataset` folder.

`-dir`: Specify the path to the directory containing document-to-topic distribution files.

`-prob`: Specify a document-to-topic distribution file OR a group of document-to-topic distribution files in the specified directory.

**Examples:**

	$ java -jar jar/STTM.jar -model ClusteringEval -label dataset/corpus.LABEL -dir results -prob corpusBTM.theta

### Step 3: interpret


## Concluding remark

Some topic models are very slow to compute! Depending on the size of your data $n$ and your vector size $K$, you may have to wait up to several days before convergence. Reasonable parameters (i.e. reasonable computing time and space) are $K <= 500$ and $n <= 10^6$.
