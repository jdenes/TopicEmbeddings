# A function to check required packages and install them 
loadPackages <- function() {
	packages <- c("data.table", "stm", "tm", "topicmodels")
    pkgcheck <- match( packages, utils::installed.packages()[,1] )
    pkgtoinstall <- packages[is.na(pkgcheck)]
    if (length(pkgtoinstall) > 0) {
		print("Installing required R packages...")
		utils::install.packages(pkgtoinstall, repos="https://cran.univ-paris1.fr/")
	}
    for (pkg in packages) {
		suppressPackageStartupMessages(library(pkg, character.only=TRUE, quietly=TRUE))
	}
}

# A text processing custom function
processText <- function(documents,
                        lowercase=TRUE, removestopwords=TRUE, removenumbers=TRUE, removepunctuation=TRUE, stem=TRUE, 
                        wordLengths=c(3,Inf),sparselevel=1, language="french",
                        verbose=TRUE, onlycharacter=FALSE,striphtml=FALSE,
                        customstopwords=NULL, v1=FALSE) {
  
  documents <- as.character(documents)
  
  if(striphtml) documents <- gsub('<.+?>', ' ', documents)
  #remove non-visible characters
  documents <- stringr::str_replace_all(documents,"[^[:graph:]]", " ")
  if(onlycharacter) documents <- gsub("[^[:alnum:]///' ]", " ", documents)
  if(verbose) cat("Building corpus... \n")
  txt <- tm::VCorpus(tm::VectorSource(documents), readerControl=list(language= language))
  #Apply filters
  txt <- tm::tm_map(txt, tm::stripWhitespace)
  if(lowercase){
    if(verbose) cat("Converting to Lower Case... \n")
    if(utils::packageVersion("tm") >= "0.6") {
      txt <- tm::tm_map(txt, tm::content_transformer(tolower)) 
    } else {
      txt <- tm::tm_map(txt, tolower)
    }
  }
  if(!v1) {
    if(removepunctuation){
      if(verbose) cat("Removing punctuation... \n")
      txt <- tm::tm_map(txt, tm::removePunctuation, preserve_intra_word_dashes = TRUE) #Remove punctuation
    }
  }
  if(removestopwords){
    if(verbose) cat("Removing stopwords... \n")
    txt <- tm::tm_map(txt, tm::removeWords, tm::stopwords(language)) #Remove stopwords
  }
  if(!is.null(customstopwords)) {
    if(verbose) cat("Remove Custom Stopwords...\n")
    txt <- tm::tm_map(txt, tm::removeWords, customstopwords)
  }
  if(removenumbers){
    if(verbose) cat("Removing numbers... \n")
    txt <- tm::tm_map(txt, tm::removeNumbers) #Remove numbers
  }
  if(v1) {
    #return to the v1 style of removing punctuation right before stemming
    if(removepunctuation){
      if(verbose) cat("Removing punctuation... \n")
      txt <- tm::tm_map(txt, tm::removePunctuation, preserve_intra_word_dashes = TRUE) #Remove punctuation
    }    
  }
  if(stem){
    if(verbose) cat("Stemming... \n")
    txt <- tm::tm_map(txt, tm::stemDocument, language=language)
  }
  #Make a matrix
  if(verbose) cat("Creating Output... \n")
  dtm <- tm::DocumentTermMatrix(txt, control=list(wordLengths=wordLengths))
  return(dtm)
}