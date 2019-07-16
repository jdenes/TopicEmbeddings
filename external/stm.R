library(stm)
library(tm)
library(data.table)

source("utils.R")

########################## PREPROCESSING ##########################

# Set main values
vector_size = 200

# Load data
data = read.csv("../datasets/INTERVENTIONS_PARTICIPANTS_2.csv", header=TRUE, encoding="UTF-8", stringsAsFactors=FALSE)
data$year = as.numeric(data$year)
rownames(data) = 1:nrow(data)

###################### STRUCTURAL TOPIC MODEL ######################

processed = textProcessor(data$res, metadata=data, language='french')
out = prepDocuments(processed$documents, processed$vocab, processed$meta)

# Build Structural Topic Model (STM)
model = stm(documents=out$documents, vocab=out$vocab, K=vector_size, max.em.its=200, data=out$meta, init.type="Spectral", seed=123)

proba = make.dt(model)
data = data[proba$docnum,]
proba$docnum = NULL
data = cbind(data, proba)

con = file(paste0('../encodings/RADIO_STM_ENCODING_', vector_size, '.csv'), encoding="UTF-8")
write.csv(data, file=con, row.names=FALSE)
sink(paste0('../encodings/RADIO_STM_WORDS_', vector_size, '.txt'))
labelTopics(model, 1:vector_size, 10)
sink()
