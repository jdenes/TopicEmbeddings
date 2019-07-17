#! /usr/bin/Rscript

library(topicmodels)
library(data.table)
source("./external/utils.R")

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("Three arguments must be supplied: input (.csv) file, K (vector size) and language.", call.=FALSE)
} else {
  filepath = args[1]
  K = args[2]
  language = args[3]
}

########################## PREPROCESSING ##########################

# Load data
data = read.csv(filepath, header=TRUE, encoding="UTF-8", stringsAsFactors=FALSE)
data$year = as.numeric(data$year)
rownames(data) = 1:nrow(data)

###################### CORRELATED TOPIC MODEL ######################

corpus = processText(data$text, language=language)
model1 = CTM(x=corpus, k=K, method="VEM", control=list(verbose=1, seed=123))

proba1 = as.data.frame(model1@gamma)
data = data[rownames(proba1),]

con = file(paste0('./external/raw_embeddings/tmp_CTM_EMBEDDING_', K, '.csv'), encoding="UTF-8")
write.table(proba1, file=con, row.names=FALSE, col.names=FALSE)
con = file(paste0('./external/raw_embeddings/tmp_CTM_LABELS_', K, '.csv'), encoding="UTF-8")
write.table(data$label, file=con, row.names=FALSE, col.names=FALSE)
sink(paste0('./external/raw_embeddings/tmp_CTM_WORDS_', K, '.txt'))
terms(model1, 20)
sink()
