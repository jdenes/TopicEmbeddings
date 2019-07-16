# Topic Embeddings Framework
A code to create and test document embeddings using topic models. Based on forthcoming paper (maybe): "Document Embedding Using Topic Models", a Master Thesis at Sorbonne Université.

Topic Embeddings Framework is a [Python 3](https://www.python.org/downloads/) open-source code to generate and test document embeddings using most common topic models. It is designed to allow easier and centralized generation of such embeddings, and to test their performances against mainstream embeddings such as bag of word or doc2vec.

**Author :** [Julien Denes](https://github.com/jdenes/), Sciences Po médialab.

---

## Algorithms

### Included

* List of the
* algorithms implemented
* include references
* DTM: input has to be a .csv file with a column called `year` to perform time slicing (see DTM architecture for better understanding). Also the file needs to be sorted according to this column.

### Supported

* Oopsi not all of the are implemented
* Here are some we currently support but need precomputation
* The list
* See [Usage](#Usage) to get how to use both

## Input data

You can use either your data or 20News in order to perform classification and assess performances.

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

### Step 2: use your embedding(s) on a classification task and interpret

**Additional input:** your file with labels.

**Usage:**

	$ python classify.py –model True –label <label_file_path> -prob <Document-topic-prob/Suffix>

`–label`: Specify the path to the ground truth label file. Each line in this label file contains the golden label of the corresponding document in the input corpus. See files `corpus.LABEL` and `corpus.txt` in the `dataset` folder.

`-dir`: Specify the path to the directory containing document-to-topic distribution files.

`-prob`: Specify a document-to-topic distribution file OR a group of document-to-topic distribution files in the specified directory.

**Examples:**

	$ java -jar jar/STTM.jar -model ClusteringEval -label dataset/corpus.LABEL -dir results -prob corpusBTM.theta
