# ++++++++++++++++++++++++++++
# dataCleaning
# ++++++++++++++++++++++++++++
# datset : data matrix from the kreditechDSTest2017 dataset with or without labels
# performs the whole data preprocessing identified during my investigation
dataCleaning <- function(datset){
  require(DMwR)
  require(dplyr)
  require(stats)
  datset$v9 <- as.factor(datset$v9)
  datset$v24 <- as.factor(datset$v24)
  datset <- select(datset, -c(v42, v95))
  dat.num.cols <- datset[, which(sapply(datset, is.numeric))]
  dat.chr.cols <- datset[, -which(sapply(datset, is.numeric))]
  dat.num.cols <- knnImputation(dat.num.cols[, ]) 
  #dat.chr.cols %>% mutate_all(as.factor)  # doesn't work yet
  datset <- bind_cols(dat.num.cols, dat.chr.cols)
  datset <- datset[complete.cases(datset),]
  return(datset)
}

#########test##########

#setwd("/home/manniv/R/kreditechDSTest2017")
#training.data <- read.csv2("Data/Training.csv", header = TRUE)
#validation <- read.csv2("Data/Validation.csv", header = TRUE)
#validation <- bind_rows(validation, training.data)
#list.obj <- dataCleaning(datset = validation)
#library(mice)
#anyNA(list.obj)
#list.obj %>% mutate_all(as.factor)
#str(list.obj)
