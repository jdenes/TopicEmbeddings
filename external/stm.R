source("./external/utils.R")
loadPackages()

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("Three arguments must be supplied: input (.csv) file, K (vector size) and language.", call.=FALSE)
} else {
  filepath = args[1]
  K = as.numeric(args[2])
  language = args[3]
}

# Load data
data = read.csv(filepath, header=TRUE, encoding="UTF-8", stringsAsFactors=FALSE)
rownames(data) = 1:nrow(data)
label = data$label
data$label = NULL

# Pre-process
processed = textProcessor(data$text, metadata=data, language=language)
out = prepDocuments(processed$documents, processed$vocab, processed$meta)

# Build Structural Topic Model (STM)
model = stm(documents=out$documents, vocab=out$vocab, K=K, max.em.its=500, data=out$meta, init.type="Spectral", seed=123)

# restrict dataframe to non-empty docs
proba = make.dt(model)
label = label[proba$docnum]
proba$docnum = NULL

# Save results
con = file(paste0('./external/raw_embeddings/tmp_STM_EMBEDDING_', K, '.csv'), encoding="UTF-8")
write.table(proba, file=con, row.names=FALSE, col.names=FALSE)
con = file(paste0('./external/raw_embeddings/tmp_STM_LABELS_', K, '.csv'), encoding="UTF-8")
write.table(label, file=con, row.names=FALSE, col.names=FALSE)
sink(paste0('./external/raw_embeddings/tmp_STM_WORDS_', K, '.txt'))
labelTopics(model, 1:K, 10)
sink()