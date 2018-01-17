# Brief summary:

# This document contains my analytical process with the data, from data refining
# tuning to model building. If you'd just like to see the refining steps, see 
# the script I made called dataCleaning.R, found in the folder R/

# Observations; v68 correlates very highly to dependent variable, v95 has too many
# missing values to be of use, v42 is 10^4 of another variable, column types are
# set inconsistently, missing values show strong patterns but may not hold enough 
# useful data in a dataset that was obfuscated.

# Model creation was finallised after utilizing many different libraries and my own
# functions. To avoid having to install all those libraries yourself, just use the
# RDS files (classifiers rf1 and rf2). Note that both ultimately failed with the 
# validation set even after scoring (suspiciously) highly on training sets due to 
# 'variables in the training data missing in newdata'when making Random Forests
# Possibly due to the rf.train parameter creating it's own dummy variables, which 
# rf test does not do in certain situations.

# Figures that were created during analysis are all found in the Figure/ folder



setwd("/home/manniv/R/kreditechDSTest2017")
tmp.sourcedat <- read.csv2("Data/Training.csv", header = TRUE)

# set paralel processing

library(doSNOW)
cl <- makeCluster(3, type = "SOCK") #
registerDoSNOW(cl)

# split into training and test set, 70% training, same distributions
library(caret)
set.seed(3333)
tmp.intrain <- createDataPartition(y = tmp.sourcedat$classlabel, p= 0.7, list = FALSE)
tmp.train <- tmp.sourcedat[tmp.intrain,]
test <- tmp.sourcedat[-tmp.intrain,]

# test and train separated immediately

# create id row in case there are merging problems in future :/ 
# one ID is cast into numerical variables, the other into factors (covers all cols for this case)

tmp.train$ID <- seq.int(nrow(tmp.train))
tmp.train$IDfac <- tmp.train$ID
tmp.train$IDfac <- as.factor(tmp.train$IDfac)

str(tmp.train)
#v9 and v24 are actually factors, changing;
tmp.train$v9 <- as.factor(tmp.train$v9)
tmp.train$v24 <- as.factor(tmp.train$v24)
summary(tmp.train)
# a lot of columns have min and Q1 as 0, suggesting properties that many individuals
# haven't used (an optional service etc.). Should those be changed to NA?
# attempting analysis without injecting 0 = NAs for now


# quick correlation plot of numerical columns, after splitting into 
# numerical and factors
library(corrplot)
library(Hmisc)
# and load a custom function for flattening corrmatrix to 4 columns containing
# both correlation coefficients and p-values
source("R/flattenCorrMatrix.R")

# assumption; columns are either numeric or factors, which holds in this dataset
tmp.numeric.columns <- tmp.train[, sapply(tmp.train, is.numeric)]
tmp.factor.columns <- tmp.train[, names(Filter(is.factor, tmp.train))]

# create matrix and p-values for numeric columns
tmp.corr.matrix <- cor(tmp.numeric.columns, 
                       method = "pearson", 
                       use = "complete.obs")
corrplot(tmp.corr.matrix, 
         method = "number", type = "full")
tmp.sigcorr <- rcorr(as.matrix(tmp.corr.matrix))

# use flattining fuction
flattenCorrMatrix(tmp.sigcorr$r, tmp.sigcorr$P)

# Insignificant correlation are crossed, p values above 0.01 are blanked out
corrplot(tmp.sigcorr$r, type="upper", order="hclust", 
         p.mat = tmp.sigcorr$P, sig.level = 0.01, insig = "blank")

# tests identified v42 and v55 as strongly and significantly correlated
# checking what is going on there
summary(tmp.numeric.columns[, c("v42", "v55")])
# interestingly, v42 seems to be v55 x 10^4, as both are otherwise equal, one will
# be removed

# find extreme values in numeric data, without v42
tmp.numeric.columns <- tmp.numeric.columns[, -6]
boxplot(x = as.list(as.data.frame(tmp.numeric.columns[, ])))
summary(tmp.numeric.columns$v53)
# very large values for v53, with one extreme outlier at 100,000, checking everything
# greater than Q3
summary(tmp.train[which(tmp.train[,"v53"]>1065),])
# summary shows many entries over Q3, trying near max value
summary(tmp.train[which(tmp.train[,"v53"]>50000),])
# values exist upto 100,000 therefore this won't be considered an outlier

boxplot(x = as.list(
  as.data.frame(
    tmp.numeric.columns[, !names(
      tmp.numeric.columns) %in% c("v53", "ID")])))

# ... this could go on, so once again doing summary instead
summary(tmp.numeric.columns)
# all variables > approx. 50 seem to follow an exponential decay. Outliers 
# therefore unlikely


# quick check for missing values
library(Amelia)
missmap(tmp.train, main = "Missing values vs observed")
# map shows many missing values in v95, with decreasing number of missing
# values till IDfac. v95 will not be used for analysis as over 50% is missing
# horizontal white bars observed, but those must be graphical errors as ID has 
# no missing data

# use second library to double check pattern on missing data
library(mice)
md.pattern(tmp.train)
# shows no entries with near empty rows. 1026 entries have all data, 1143 entries
# are only missing v95, others seem to be almost evenly dispersed 
# as removing v95 gives 2169 complete datasets, or 83% of the total training data
# the remaining are imputed through KNN

# use K nearest neighbors to impute missing values for the numeric data
# assumes MNAR- missing not at random
# factor variables are not used as they can heavily influence KNN. As I don't
# know at this point if there is a highly significant factor variable, I'm
# leaving this out

library(DMwR)
tmp.noNAs.knn.numeric <- knnImputation(tmp.numeric.columns[, ]) 
anyNA(tmp.noNAs.knn.numeric)
# check to see if it worked. anyNA produced FALSE
# safe to use as investigated label 'classlabel' is a factor and not present

# comparing new to old data
summary(tmp.noNAs.knn.numeric)
summary(tmp.numeric.columns)
# max and Q3 values for columns with imputted data didn't change. Means and Q1s
# changed slightly. Seems ok

# add non numeric columns back
tmp.factor.columns$IDfac <- as.numeric(tmp.factor.columns$IDfac)
tmp.noNAs.knn.numeric$ID <- as.numeric(tmp.noNAs.knn.numeric$ID)

library(dplyr)
training <- bind_cols(tmp.factor.columns, tmp.noNAs.knn.numeric)

# check if rows matched correctly through use of IDs
identical(training[['ID']],training[['IDfac']])
# great, now removing IDs, v95 as they hold no more value or have many missing vals
training <- select(training, -c(ID, IDfac, v95))
# checking with str shows new training set has all columns -c(v95, v42) as test did, with
# same number of entries (rows). Preprocessing almost done.


# reset global environment by deleting temporary variables
rm(list = ls(pattern = "^tmp"))

# checking missing values one final time, as numeric columns are now combined
# with uncleaned categorical columns
md.pattern(training)
# still shows 44 cases with v32, v85, v33 and v99 missing, which may be a strong
# feature for feature creation. As context doesn't exist for this dataset,
# I've decided to remove incomplete entries as they make 75/2516 or ~ 3% of all 
# entries.

library(stats)
training <- training[complete.cases(training),]
anyNA(training)
# all remaining NAs removed

# using tSNE to further reduce dimensionality to 2d space
library(Rtsne)
tmp.tsne.1 <- Rtsne(training[, ], check_duplicates = FALSE)
tmp.indexes <- which(training$classlabel != "None")
tsne.1 <- tmp.tsne.1
tsne.indexes <- tmp.indexes

ggplot(NULL, aes(x = tmp.tsne.1$Y[tmp.indexes, 1], y = tmp.tsne.1$Y[tmp.indexes, 2], 
                 color = training$classlabel[tmp.indexes])) +
  geom_point() +
  labs(color = "Yes./No.") +
  ggtitle("tsne 2D Visualization of Features")         

# tSNE produced well defined clusters but shows little separation between labels yes and no
# initially using raw table data to make RF

############################models###############################
# chose random forest as data is a mix of variably ordered factors and numeric
# variables. Risk exists in that weighted RandomForests favor high order factors
# if individual decision trees show bias towards factors like v24 or v19, one hot
# encoding might be necessary for a NN.

rm(list = ls(pattern = "^tmp"))

library(randomForest)
# Train a Random Forest with the default parameters using significant variables
# paralelization already set
tmp.rf.train1 <- select(training, -c(classlabel))
tmp.rf.train.label1 <- as.factor(training$classlabel)

tmp.rf.1 <- randomForest(x = tmp.rf.train1, y = tmp.rf.train.label1, importance = TRUE, ntree = 1000)
rf.1 <- tmp.rf.1
# baseline randomforest gives us an out of box error rate of 0.12%... suspicously low?
rm(list = ls(pattern = "^tmp"))

# double check with test set
# created separate function to repeat data cleaning
# call function dataCleaning to get data with imputed numerical values and
# removed NA factor values, as well as removal of highly correlating columns
source("R/dataCleaning.R")
tmp.clean.test <- dataCleaning(test)
anyNA(tmp.clean.test)
tmp.rf.test1 <- select(tmp.clean.test, -c(classlabel))
tmp.rf.test.label1 <- as.factor(tmp.clean.test$classlabel)

# make predictions on test set
tmp.rf.1.preds <- predict(rf.1, tmp.rf.test1)

# calculate accuracy (rough value, no inspection of likelyhood i.e. AUROC)
tmp.correct.ans <- tmp.rf.1.preds == tmp.rf.test.label1
tmp.t<-table(tmp.correct.ans, tmp.rf.test.label1)
print(tmp.t)
# accuracy was at (1000/1080) i.e. at 92%. At such high success rates, it's hard
# to define where overfitting starts but as this was unseen data, it's encouraging

# plot variable importance, clearly v68 had a very strong correlation with
# the dependant variable
varImpPlot(rf.1)

# clear everything
rm(list = ls(pattern = "^tmp"))
### end of creating model ###


####### final predictions: #######

# validation set seems to be formatted differently to training set
# to remedy this, train, test and validation sets are combined, processed together
# in dataCleaning, a new RF is trained before evaluating with the final evaluation
# set

tmp.validation <- read.csv2("Data/Validation.csv", header = TRUE)
tmp.training <- read.csv2("Data/Training.csv", header = TRUE)

validation <- bind_rows(tmp.validation, tmp.training)
#warnings()
# all data now in uniformly typed columns

# clean all data using dataCleaning again
source("R/dataCleaning.R")
tmp.clean.valid <- dataCleaning(validation)
anyNA(tmp.clean.valid)

# try convert chr columns to factor
# yes ugly workaround till i figure out how mutate_all works for my dataCleaning func.
tmp.clean.valid$v33 <- as.factor(tmp.clean.valid$v33)
tmp.clean.valid$v32 <- as.factor(tmp.clean.valid$v32)
tmp.clean.valid$v99 <- as.factor(tmp.clean.valid$v99)
tmp.clean.valid$v85 <- as.factor(tmp.clean.valid$v85)

# slice data into training set (3591 training data)
tmp.clean.training <- slice(tmp.clean.valid, 1:3591)
tmp.rf.training <- select(tmp.clean.training, -c(classlabel))
tmp.rf.training.label1 <- as.factor(tmp.clean.training$classlabel)

# now train a new TF model with large training data
tmp.rf.2 <- randomForest(x = tmp.clean.training, y = tmp.rf.training.label1, importance = TRUE, ntree = 1000)
rf.2 <- tmp.rf.2


# recover last 200 values as they are from the validation file
tmp.clean.valid <- slice(tmp.clean.valid, 3592:3791)
# re-create validation set
tmp.rf.valid <- select(tmp.clean.valid, -c(classlabel))
tmp.rf.valid.label1 <- as.factor(tmp.clean.valid$classlabel)
# make predictions on test set
tmp.rf.2.preds <- predict(rf.2, tmp.rf.valid)
# didn't work, although I controlled for column headings and types. Will need to
# do further investigation to figure out why the final test didn't work.


#Shutdown cluster
stopCluster(cl)

# final step, production of predictive model and export for use;
# p.s. modified training data will have to be available along with compressed file
saveRDS(rf.1, file = "classifier_rf1", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)
saveRDS(rf.2, file = "classifier_rf2", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)
