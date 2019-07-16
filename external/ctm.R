library(topicmodels)
library(data.table)

source("utils.R")

########################## PREPROCESSING ##########################

# Set main values
vector_size = 200

# Load data
data = read.csv("../datasets/INTERVENTIONS_PARTICIPANTS_2.csv", header=TRUE, encoding="UTF-8", stringsAsFactors=FALSE)
data$year = as.numeric(data$year)
rownames(data) = 1:nrow(data)

###################### CORRELATED TOPIC MODEL ######################

corpus = processText(data$res, language='french')
model1 = CTM(x=corpus, k=vector_size, method="VEM", control=list(verbose=1, seed=123))

proba1 = as.data.frame(model1@gamma)
data = data[rownames(proba1),]
data = cbind(data, proba1)

con = file(paste0('../encodings/RADIO_CTM_ENCODING_', vector_size, '.csv'), encoding="UTF-8")
write.csv(data, file=con, row.names=FALSE)
sink(paste0('../encodings/RADIO_CTM_WORDS_', vector_size, '.txt'))
terms(model1, 20)
sink()

