# Topic Embeddings Framework

Topic Embeddings Framework is a Python 3 open-source code to generate and test document embeddings using most common topic models. It is designed to allow easier and centralized generation of such embeddings, and to test their performances against mainstream embeddings such as bag-of-words, doc2vec or BERT. The approach is detailed on the paper ["Document Embeddings Using Topic Models"](./report/TopicEmbeddings.pdf), a Master's Thesis at Sorbonne Université.

**General idea:** the document embeddings we propose, called topic embeddings, use one particular output of any trained topic models, know as topics proportion. It provides for each document a unit vector where the ith component indicates "how much" the document relates to topic i. We test the simple hypothesis that such vector is an insightful representation of the document for various NLP tasks.


**Author :** [Julien Denes](https://github.com/jdenes/), [Sciences Po médialab](https://github.com/medialab).

---

## Algorithms

Several document embeddings are available for test in this framework. Some are built-in and very easy to use, and others need preliminary work from the user.

### Built-in

**Reference embeddings:** standard models bag-of-words ([BoW](https://doi.org/10.1080/00437956.1954.11659520)) with TF-IDF, [doc2vec](https://arxiv.org/abs/1405.4053) and Latent Semantic Analysis ([LSA](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9)) are proposed using [gensim](https://radimrehurek.com/gensim/)'s implementation. The framework also includes Word Embedding Aggregation (abreviated Pool), which represents a document as the aggregation of its words representations, using gensim's [word2vec](https://arxiv.org/abs/1310.4546) and mean as pooling function. Finally Bag of Random  Embedding  Projection ([BoREP](https://arxiv.org/abs/1901.10444)), which randomly projects words' embeddings to create the document's one, is also proposed.

**Topic embeddings:** the most classical topic models are available to test: Latent Dirichlet Allocation ([LDA](https://dl.acm.org/citation.cfm?id=944937)), Hierarchical Dirichlet Process ([HDP](https://doi.org/10.1198/016214506000000302)), Dynamic Topic Model ([DTM](https://doi.org/10.1145/1143844.1143859)), which all rely on gensim, as well as Structural Topic Model ([STM](https://doi.org/10.1111/ajps.12103)) using R package [stm](https://cran.r-project.org/web/packages/stm/index.html) and Correlated Topic Model ([CTM](https://dl.acm.org/citation.cfm?id=2976267)) using R package [topicmodels](https://cran.r-project.org/web/packages/topicmodels/index.html).

### Supported

Unfortunatly, not all topic models have simple implementations in Python or R. We support Pseudo-Document-Based Topic Model ([PTM](https://www.kdd.org/kdd2016/subtopic/view/topic-modeling-of-short-texts-a-pseudo-document-view)), which can be computed using using [STTM](https://github.com/qiang2100/STTM). Once you've done that, simply import the topic proportion file into `external/raw_embeddings/` and name it `tmp_PTM_EMBEDDING_K.csv` where K is your vector size. Remark that any topic proportion matrix of any topic model could be integrated into this framework, so do not hesitate to suggest some!

## Requirements

* [Python 3.6+](https://www.python.org/downloads/) with packages listed in `requirements.txt`. To install them, simply run `pip install -r requirements.txt`.
* If you intend to use STM or CTM, you will also need to have [R](https://www.r-project.org/) installed. Required packages will install automatically.
* If you intend to use DTM, you need to download the binary file corresponding to your OS from [this repository](https://github.com/magsilva/dtm/tree/master/bin) and place it into `./external/dtm_bin/`.
* Please make sure that both `python` and `Rscript` commands can be run in your terminal no matter the directory you're in. If you're using Windows, consider adding them to your `PATH` environment variable ([see an example](https://datatofish.com/add-python-to-windows-path/)).

## Input data

Two possible data inputs are available for now:
* **20 Newsgroups data set**, a [popular standard data set](http://qwone.com/~jason/20Newsgroups/) for experiments in NLP. Data will be automatically imported.

* **Your own data.** If you wish to do so, you need to provide a `.csv` file with two mandatory columns: one called `text` containing your text, the other called `labels` containing the document's label in numerical format. A column called `year` can be optionally included to be used in DTM or STM, in which case your data must be sorted according to this column. All other columns will be ignored, except by STM. An example file is provided: `datasets/example.csv`.


## Usage

Our algorithm runs in three steps. First, **generate the embedding** from the data and save it. Second, **run classification tests** to assess the performance of your embeddings. Third, **interpret the results** easily. All 3 steps save files on your disk to allow you to run them separately. 

### Step 1: create embeddings from the corpus

The first steps consists in getting a vector representation of each document in your corpus. You may either use reference embeddings, or topic embeddings.

**To produce it, simply run this command in your terminal:**

	$ python main.py -mode encode -input <path/to/input> -embed <String> [-project <String>] [-k <int>] [-prep <bool>] [-langu <String>]

where parameters in `[ ]` are optional. In detail, each parameter corresponds to:

`-mode encode`: to specify that you're computing your embeddings.

`-input`: specify the path to your custom input file, or `20News` to use 20 Newsgroups.

`-embed <String>`: embedding to use. Must be one of: `BOW, DOC2VEC, POOL, BOREP, LSA, LDA, HDP, DTM, STM, CTM, PTM`.

`-project`: name of your project to find the results later. Default is current day and time.

`-k <int>`: size of your embeddings, greater than 1. Default is 200.

`-prep <bool>`: specify if you want to preprocess (i.e. lowercase, lemmatize...). Default is `True`.

`-langu <String>`: if you selected preprocessing, language to use (in english). Default is `english`.

**Examples:**

    $ python main.py -mode encode -input datasets/example.csv -embed LDA -prep True -langu french
    
    $ python main.py -mode encode -input 20News -embed STM -project 20NewsTest -k 500

Once the code is done running, you'll see a new folder in `results` having your project's name. Its subfolder `embeddings` contains the matrix of vector representation of your corpus, with one row per document (if none has been dropped) and k columns. In subfolder `models`, we store the models that were used to produce those embeddings. **Notice that you can (and should!) produce several embeddings for your data set and compare them in Step 2!** They will be all be stored in your project's folder.

Please note that some topic models have specific behaviors. STM and CTM run `R` scripts located in `external` and store temporary files in `external/raw_embeddings`. PTM requires users to pre-compute embeddings as described above.

### Step 2: use your embedding(s) on a classification task

This step uses the embedding in the classification task of your choice according to the `label` column provided in your input file (or the news group for 20 Newsgroups). Several algorithms are proposed: logistic regression (logit), Naive Baiyes (NBayes), AdaBoost (AdaB), KNN with 3 neighbors, Decision Tree (DTree), Artificial Neural Network (ANN) with 3 layers of size 1000, 500 and 100, and finaly a SVM with RBF kernel.

**To run a classification, simply use this command in your terminal:**

	$ python main.py -mode classify -input <path/to/input> -embed <String> [-project <String>] [-k <int>] [-algo <String>] [-samp <String>]

where parameters in `[ ]` are optional. Parameters `-input`, `-embed`, `-project` and `-k` should be the same as in Step 1. The new parameters corresponds to:

`-mode classify`: to specify that you're using embeddings to perform classification task.

`-algo <String>`: classifier to use. Must be one of: `LOGIT, NBAYES, ADAB, DTREE, KNN, ANN, SVM`. Default is `LOGIT`.

`-samp <String>`: sampling to use to prevent imbalanced data sets. Must be one of `OVER, UNDER, NONE`. Default is `NONE`.

**Examples:**

    $ python main.py -mode classify -input datasets/example.csv -embed LDA -algo SVM -samp OVER
    
    $ python main.py -mode classify -input 20News -embed STM -project 20NewsTest -k 500
    
This step produces new files in your project's folder, under `performances`, with various performence metrics for your embedding: accuracy, precision, recall, and F1-score. Some files are also created under `classifiers` to store the trained classifier. At this point, you are able to assess if topic embeddings perform well on your data set and task!

### Step 3: interpret

You can know use the power of interpretability of topic models to get insights on what each dimenson of your embedding represent, and which of those dimensions matter in the classification task you used. This requires of course that you use a logistic regression (`LOGIT`) as a classifier in Step 2. Please note that at this step of development of the framework, only `BOW` and `LDA` approaches can be interpreted. Other built-in topic models should follow.

**To get the interpretation, simply run this command in your terminal:**

	$ python main.py -mode interpret -input <path/to/input> -embed <BOW or LDA> [-project <String>] [-k <int>] [-prep <bool>] [-langu <String>] [-algo LOGIT] [-samp <String>]

Note that all parameters `-input`, `-embed`, `-project`, `-k`, `-prep`, `-langu`, `-algo`, and `-samp` should be the same as in Step 1 or 2. New parameter value `-mode interpret` specifies that you are performing the interpretation.

### All-at-once command

Now that you understand all 3 steps of the process, we shall reveal the command to run all three steps in a row:

	$ python main.py -mode all -input <path/to/input> -embed <String> [-project <String>] [-k <int>] [-prep <bool>] [-langu <String>] [-algo <String>] [-samp <String>]

## Concluding remark

Some topic models are very slow to compute! Depending on the size of your data N and your vector size K, you may have to wait up to several days before convergence. Reasonable parameters (i.e. reasonable computing time and space) are K < 500 and N < 10^5.
